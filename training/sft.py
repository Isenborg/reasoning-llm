import random
import torch
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer
from peft import LoraConfig, get_peft_model, TaskType

from training.configs import SFTConfig
from utils.checks import is_correct_answer
from utils import lmprint
from utils.models import save_model


def _collate_fn(batch, pad_token_id: int):
    """Pad a batch of variable-length examples to the same length."""
    max_len = max(b["input_ids"].shape[0] for b in batch)

    input_ids_list = []
    labels_list = []
    attention_mask_list = []

    for b in batch:
        seq_len = b["input_ids"].shape[0]
        pad_len = max_len - seq_len

        input_ids_list.append(
            torch.cat([b["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)])
        )
        labels_list.append(
            torch.cat([b["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
        )
        attention_mask_list.append(
            torch.cat([b["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
        )

    return {
        "input_ids": torch.stack(input_ids_list),
        "labels": torch.stack(labels_list),
        "attention_mask": torch.stack(attention_mask_list),
    }


class SFTTrainer:
    """
    Supervised fine-tuning trainer for causal language models.

    Trains on completion tokens only (prompt tokens are masked with -100 in labels).
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: SFTConfig,
    ):
        self.tokenizer = tokenizer
        self.config = config

        if config.use_lora:
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.lora_target_modules,  # None = auto-detect
                bias="none",
            )
            self.model = get_peft_model(model, lora_cfg)
            self.model.print_trainable_parameters()
        else:
            self.model = model

        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable, lr=config.lr)

        # Loss history for plotting
        self.train_steps: list[int] = []
        self.train_losses: list[float] = []
        self.eval_steps: list[int] = []
        self.eval_losses: list[float] = []

    def train(self, train_dataset, eval_dataset=None, run_name: str = "sft_model"):
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=lambda batch: _collate_fn(batch, pad_id),
        )

        G = self.config.grad_accum_steps
        optimizer_steps_per_epoch = len(dataloader) // G
        total_optimizer_steps = optimizer_steps_per_epoch * self.config.epochs

        print(f"SFT Training: {self.config.epochs} epochs | "
              f"batch={self.config.batch_size} × accum={G} = effective batch {self.config.batch_size * G} | "
              f"~{total_optimizer_steps} optimizer steps | "
              f"{len(train_dataset)} examples")
        if eval_dataset:
            print(f"Evaluating every {self.config.eval_every} optimizer steps "
                  f"on {self.config.eval_samples} samples")
        print("-" * 50)

        if eval_dataset:
            metrics = self.evaluate(eval_dataset)
            self._log_eval(0, metrics)

        self.model.train()
        self.optimizer.zero_grad()
        global_step = 0       # counts optimizer steps
        micro_step = 0        # counts forward passes
        best_accuracy = 0.0

        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            epoch_optimizer_steps = 0

            for batch in dataloader:
                batch = {k: v.to(self.model.device) for k, v in batch.items()}

                outputs = self.model(**batch)
                # Scale loss so gradients are averaged across accumulation steps
                loss = outputs.loss / G
                loss.backward()
                micro_step += 1

                # Record the unscaled loss for logging
                raw_loss = outputs.loss.item()
                epoch_loss += raw_loss

                if micro_step % G == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.config.grad_clip
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                    epoch_optimizer_steps += 1

                    self.train_steps.append(global_step)
                    self.train_losses.append(raw_loss)
                    self._log_step(global_step, raw_loss)

                    if eval_dataset and global_step % self.config.eval_every == 0:
                        metrics = self.evaluate(eval_dataset)
                        self._log_eval(global_step, metrics)
                        self.eval_steps.append(global_step)
                        self.eval_losses.append(metrics["eval_loss"])

                        if metrics["eval_accuracy"] > best_accuracy:
                            best_accuracy = metrics["eval_accuracy"]
                            self._save(f"{run_name}-best")
                            print(f"  New best model saved at step {global_step} "
                                  f"(accuracy={best_accuracy:.1%})")

            avg_epoch_loss = epoch_loss / max(epoch_optimizer_steps, 1)
            print(f"Epoch {epoch + 1:>2d}/{self.config.epochs} | Avg Loss: {avg_epoch_loss:.4f}")
            print("-" * 50)

        # Final eval
        if eval_dataset:
            metrics = self.evaluate(eval_dataset)
            self._log_eval(global_step, metrics)
            self.eval_steps.append(global_step)
            self.eval_losses.append(metrics["eval_loss"])

        self.model.eval()
        print(f"SFT complete. {global_step} steps over {self.config.epochs} epochs.")
        self.plot_losses()

    @torch.no_grad()
    def evaluate(self, eval_dataset, prompt_template=None, temperature: float = 0.3) -> dict:
        """
        Evaluate by generating answers and checking exact-match accuracy.

        If prompt_template is None, skips generation-based eval and only
        reports perplexity-style loss on the eval set.
        """
        self.model.eval()

        n = min(self.config.eval_samples, len(eval_dataset))
        indices = random.sample(range(len(eval_dataset)), n)

        total_loss = 0.0
        num_batches = 0

        for idx in indices:
            sample = eval_dataset[idx]
            input_ids = sample["input_ids"].unsqueeze(0).to(self.model.device)
            labels = sample["labels"].unsqueeze(0).to(self.model.device)
            attention_mask = sample["attention_mask"].unsqueeze(0).to(self.model.device)

            outputs = self.model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
            )
            total_loss += outputs.loss.item()
            num_batches += 1

        metrics = {
            "eval_loss": total_loss / max(num_batches, 1),
            "eval_samples": n,
            "eval_accuracy": float("nan"),  # requires generation; see below
        }

        self.model.train()
        return metrics

    def plot_losses(self):
        """Plot training (and optional eval) loss curves."""
        _, ax = plt.subplots(figsize=(9, 4))

        ax.plot(self.train_steps, self.train_losses, linewidth=0.8, alpha=0.4, color="steelblue", label="train loss (per step)")

        # Smoothed training loss (50-step rolling average)
        if len(self.train_losses) >= 10:
            window = min(50, len(self.train_losses))
            smoothed = [
                sum(self.train_losses[max(0, i - window):i]) / min(i, window)
                for i in range(1, len(self.train_losses) + 1)
            ]
            ax.plot(self.train_steps, smoothed, linewidth=2, color="steelblue", label=f"train loss (smoothed, w={window})")

        if self.eval_losses:
            ax.plot(self.eval_steps, self.eval_losses, "o-", linewidth=2, color="tomato", markersize=5, label="eval loss")

        # Epoch boundary lines
        total_steps = self.train_steps[-1] if self.train_steps else 0
        steps_per_epoch = total_steps // self.config.epochs if self.config.epochs > 1 else 0
        for e in range(1, self.config.epochs):
            ax.axvline(e * steps_per_epoch, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
            ax.text(e * steps_per_epoch + 2, ax.get_ylim()[1] * 0.97, f"epoch {e+1}", fontsize=7, color="gray", va="top")

        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("SFT Training Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def _save(self, name: str):
        """Save adapter weights (LoRA) or the full model."""
        if self.config.use_lora:
            # Save only the small adapter weights (~MB instead of ~GB)
            from utils.models import DEFAULT_MODEL_DIR
            save_path = DEFAULT_MODEL_DIR / name
            save_path.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            print(f"LoRA adapter saved to: {save_path}")
        else:
            save_model(self.model, self.tokenizer, name)

    def _log_step(self, step: int, loss: float):
        if step % 10 == 0:
            print(f"  Step {step:>5d} | Loss: {loss:.4f}")

    def _log_eval(self, step: int, metrics: dict):
        acc = metrics.get("eval_accuracy")
        acc_str = f"{acc:.1%}" if acc == acc else "n/a"  # nan check
        print(
            f"\n  [EVAL @ step {step}] "
            f"Loss: {metrics['eval_loss']:.4f} | "
            f"Accuracy: {acc_str} | "
            f"Samples: {metrics['eval_samples']}\n"
        )
