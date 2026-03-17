import time
import random
import torch
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer, get_cosine_schedule_with_warmup


from training.configs import SFTConfig
from utils.lora import apply_lora, merge_lora
from utils.checks import is_correct_answer
from utils import lmprint
from utils.models import save_model
from grpo.utils import generate_prompt, generate_rollouts
import bitsandbytes as bnb


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
            self.model = apply_lora(
                model,
                r=config.lora_r,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
                target_modules=config.lora_target_modules,
            )
        else:
            self.model = model

        # Speedups
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.model = torch.compile(self.model)

        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if config.use_8bit_optim:
            self.optimizer = bnb.optim.AdamW8bit(trainable, lr=config.lr)
        else:
            self.optimizer = AdamW(trainable, lr=config.lr)

        # Scheduler created later in train() once we know total steps
        self.scheduler = None

        # Loss history for plotting
        self.train_steps: list[int] = []
        self.train_losses: list[float] = []
        self.eval_steps: list[int] = []
        self.eval_losses: list[float] = []
        self._step_times = []

    def train(self, train_dataset, eval_dataset=None, run_name: str = "sft_model"):
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=lambda batch: _collate_fn(batch, pad_id),
            num_workers=4,
            pin_memory=True,
        )

        G = self.config.grad_accum_steps
        optimizer_steps_per_epoch = len(dataloader) // G
        total_optimizer_steps = optimizer_steps_per_epoch * self.config.epochs


        # Create LR scheduler
        warmup_steps = int(0.03 * total_optimizer_steps)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_optimizer_steps,
        )

        print(f"SFT Training: {self.config.epochs} epochs | "
              f"batch={self.config.batch_size} × accum={G} = effective batch {self.config.batch_size * G} | "
              f"~{total_optimizer_steps} optimizer steps | "
              f"{len(train_dataset)} examples | "
              f"LR: {self.config.lr} with cosine schedule ({warmup_steps} warmup steps)")
        
        if eval_dataset:
            print(f"Evaluating every {self.config.eval_every} optimizer steps "
                  f"on {self.config.eval_samples} samples")
        print("-" * 50)

        if eval_dataset:
            metrics = self.evaluate(eval_dataset)
            self._log_eval(0, metrics)

        self.model.train()
        self.optimizer.zero_grad()
        global_step = 0
        micro_step = 0
        best_eval_loss = float("inf")
        accum_loss = 0.0

        # Throughput tracking
        step_tokens = 0
        step_start = time.time()
        total_tokens = 0
        train_start = time.time()

        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            epoch_optimizer_steps = 0

            for batch in dataloader:
                batch = {k: v.to(self.model.device) for k, v in batch.items()}

                # Count real (non-padding) tokens in this micro-batch
                batch_tokens = batch["attention_mask"].sum().item()
                step_tokens += batch_tokens
                total_tokens += batch_tokens

                outputs = self.model(**batch)
                loss = outputs.loss / G
                loss.backward()
                micro_step += 1

                # Accumulate unscaled loss across micro-batches
                accum_loss += outputs.loss.item()

                if micro_step % G == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.config.grad_clip
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                    epoch_optimizer_steps += 1

                    # Average loss over all micro-batches in this step
                    avg_loss = accum_loss / G
                    accum_loss = 0.0

                    epoch_loss += avg_loss

                    # Calculate tokens/sec for this optimizer step
                    step_elapsed = time.time() - step_start
                    tok_per_sec = step_tokens / step_elapsed if step_elapsed > 0 else 0

                    # ETA based on elapsed time and step progress
                    elapsed = time.time() - train_start
                    remaining_steps = total_optimizer_steps - global_step
                    # Moving average step time
                    self._step_times.append(step_elapsed)
                    window = self._step_times[-50:]
                    sec_per_step = sum(window) / len(window)
                    
                    eta_seconds = remaining_steps * sec_per_step

                    self.train_steps.append(global_step)
                    self.train_losses.append(avg_loss)
                    self._log_step(global_step, total_optimizer_steps, avg_loss, tok_per_sec, eta_seconds)

                    # Reset for next optimizer step
                    step_tokens = 0
                    step_start = time.time()

                    if eval_dataset and global_step % self.config.eval_every == 0:
                        metrics = self.evaluate(eval_dataset)
                        self._log_eval(global_step, metrics)
                        self.eval_steps.append(global_step)
                        self.eval_losses.append(metrics["eval_loss"])

                        # Reset timer so eval time isn't counted
                        step_start = time.time()

            avg_epoch_loss = epoch_loss / max(epoch_optimizer_steps, 1)
            print(f"Epoch {epoch + 1:>2d}/{self.config.epochs} | Avg Loss: {avg_epoch_loss:.4f}")
            print("-" * 50)
            self._save(run_name + "-epoch-" + str(epoch + 1))


        # Final eval
        if eval_dataset:
            metrics = self.evaluate(eval_dataset)
            self._log_eval(global_step, metrics)
            self.eval_steps.append(global_step)
            self.eval_losses.append(metrics["eval_loss"])

        # Overall throughput
        total_elapsed = time.time() - train_start
        avg_tok_per_sec = total_tokens / total_elapsed if total_elapsed > 0 else 0

        self.model.eval()
        print(f"SFT complete. {global_step} steps over {self.config.epochs} epochs.")
        print(f"Total tokens: {total_tokens:,} | "
              f"Time: {self._format_eta(total_elapsed)}")
        if self.config.plot_training:
            self.plot_losses()

        self._save(run_name)

    @torch.no_grad()
    def evaluate(self, eval_dataset, generate: bool = False) -> dict:
        """
        Loss eval (fast, every N steps) + optional generation eval (slow, end of epoch).
        """
        self.model.eval()

        n = min(self.config.eval_samples, len(eval_dataset))
        indices = random.sample(range(len(eval_dataset)), n)

        # 1. Teacher-forced loss (always)
        total_loss = 0.0
        for idx in indices:
            sample = eval_dataset[idx]
            input_ids = sample["input_ids"].unsqueeze(0).to(self.model.device)
            labels = sample["labels"].unsqueeze(0).to(self.model.device)
            attention_mask = sample["attention_mask"].unsqueeze(0).to(self.model.device)

            outputs = self.model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
            total_loss += outputs.loss.item()

        metrics = {
            "eval_loss": total_loss / n,
            "eval_samples": n,
        }

        self.model.train()
        return metrics

    def plot_losses(self, save_path: str = "sft_training_loss.png"):
        """Plot training (and optional eval) loss curves and save to file."""
        fig, ax = plt.subplots(figsize=(9, 4))

        ax.plot(self.train_steps, self.train_losses, linewidth=0.8, alpha=0.4,
                color="steelblue", label="train loss (per step)")

        # Smoothed training loss (50-step rolling average)
        if len(self.train_losses) >= 10:
            window = min(50, len(self.train_losses))
            smoothed = [
                sum(self.train_losses[max(0, i - window):i]) / min(i, window)
                for i in range(1, len(self.train_losses) + 1)
            ]
            ax.plot(self.train_steps, smoothed, linewidth=2, color="steelblue",
                    label=f"train loss (smoothed, w={window})")

        if self.eval_losses:
            ax.plot(self.eval_steps, self.eval_losses, "o-", linewidth=2,
                    color="tomato", markersize=5, label="eval loss")

        # Epoch boundary lines
        total_steps = self.train_steps[-1] if self.train_steps else 0
        steps_per_epoch = total_steps // self.config.epochs if self.config.epochs > 1 else 0
        for e in range(1, self.config.epochs):
            ax.axvline(e * steps_per_epoch, color="gray", linestyle="--",
                    linewidth=0.8, alpha=0.6)
            ax.text(e * steps_per_epoch + 2, ax.get_ylim()[1] * 0.97,
                    f"epoch {e+1}", fontsize=7, color="gray", va="top")

        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("SFT Training Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Training plot saved to {save_path}")

    def _save(self, name: str):
        """Save the model. If LoRA was used, merge first, then save full model."""
        from utils.models import save_model

        model = self.model
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod

        if self.config.use_lora:
            merge_lora(model)

        save_model(model, self.tokenizer, name)
    def _log_step(self, step, total_steps, loss, tok_per_sec, eta_seconds):
        if step % 10 == 0:
            current_lr = self.scheduler.get_last_lr()[0]
            print(f"  Step {step:>5d}/{total_steps} | Loss: {loss:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"{tok_per_sec:,.0f} tok/s | ETA: {self._format_eta(eta_seconds)}")

    def _log_eval(self, step: int, metrics: dict):
        print(
            f"\n  [EVAL @ step {step}] "
            f"Loss: {metrics['eval_loss']:.4f} | "
            f"Samples: {metrics['eval_samples']}\n"
        )

    def _format_eta(self, seconds: float) -> str:
        """Format seconds into a human-readable string."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            m, s = divmod(seconds, 60)
            return f"{int(m)}m {int(s)}s"
        else:
            h, remainder = divmod(seconds, 3600)
            m, s = divmod(remainder, 60)
            return f"{int(h)}h {int(m)}m {int(s)}s"