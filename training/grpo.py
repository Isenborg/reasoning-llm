import torch
import math
import random
import wandb
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer
import time

from grpo.functions import (
    get_per_token_logps,
    compute_advantages,
    grpo_loss,
)
from grpo.utils import generate_rollouts
from training.configs import GRPOConfig
from utils.models import save_model
from utils.checks import is_correct_answer
from utils import lmprint
import bitsandbytes as bnb

# Lower matmul precision
torch.set_float32_matmul_precision("high")


class GRPOTrainer:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        reward_fn,
        config: GRPOConfig,
    ):
        self.model = model
        self.model.gradient_checkpointing_enable()
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.config = config

        if self.config.use_8bit_optim:
            self.optimizer = bnb.optim.AdamW8bit(
                self.model.parameters(), lr=self.config.lr
            )
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.config.lr
            )

        self.timings = {
            "generation": 0.0,
            "training": 0.0,
            "total_gen_tokens": 0,
            "total_train_tokens": 0,
        }

    def _compute_rewards(
        self, response_texts: list[str], ground_truth: str
    ) -> torch.Tensor:
        rewards = [self.reward_fn(resp, ground_truth) for resp in response_texts]
        return torch.tensor(rewards, dtype=torch.float, device=self.model.device)

    @torch.no_grad()
    def _cache_logps(self, rollout):
        logps = get_per_token_logps(
            self.model,
            rollout["input_ids"],
            rollout["attention_mask"],
        )
        return logps

    def _train_step(self, rollout, old_logps, advantages, n_rollouts):
        G = rollout["input_ids"].shape[0]
        total_loss = 0.0

        for i in range(G):
            ids = rollout["input_ids"][i : i + 1]
            mask = rollout["attention_mask"][i : i + 1]
            r_mask = rollout["response_mask"][i : i + 1]
            old_lp = old_logps[i : i + 1]
            adv = advantages[i : i + 1]

            new_lp = get_per_token_logps(self.model, ids, mask)
            log_ratio = new_lp - old_lp
            ratios = torch.exp(log_ratio)

            loss = grpo_loss(ratios, adv, r_mask, epsilon=self.config.epsilon)
            scaled = loss / (G * n_rollouts)
            scaled.backward()

            total_loss += loss.item()

        return total_loss / G

    def _compute_batch_metrics(self, rollouts, all_rewards):
        """Extract useful metrics from training rollouts."""
        all_r = torch.cat(all_rewards)

        lengths = []
        for rollout in rollouts:
            lens = rollout["response_mask"].sum(dim=-1)
            lengths.append(lens)
        all_lengths = torch.cat(lengths).float()

        return {
            "train/mean_reward": all_r.mean().item(),
            "train/reward_std": all_r.std().item(),
            "train/fraction_correct": (all_r > 0.5).float().mean().item(),
            "train/mean_response_length": all_lengths.mean().item(),
            "train/max_response_length": all_lengths.max().item(),
            "train/min_response_length": all_lengths.min().item(),
        }

    @torch.no_grad()
    def evaluate(self, eval_dataset):
        self.model.eval()

        n_total = min(self.config.eval_samples, len(eval_dataset))
        indices = random.sample(range(len(eval_dataset)), n_total)
        batch_size = 32

        total_reward = 0.0
        correct_count = 0
        total_tokens = 0.0

        printed_correct = False
        printed_incorrect = False

        for i in range(0, n_total, batch_size):
            batch_indices = indices[i : i + batch_size]
            batch_examples = [eval_dataset[idx] for idx in batch_indices]

            qs = [ex["question"] for ex in batch_examples]
            ans = [ex["answer"] for ex in batch_examples]

            rollouts = generate_rollouts(
                model=self.model,
                tokenizer=self.tokenizer,
                questions=qs,
                G=1,
                max_new_tokens=self.config.max_new_tokens,
                temperature=0.3,
            )

            for j, rollout in enumerate(rollouts):
                resp = rollout["response_texts"][0]
                ground_truth = ans[j]

                is_corr = is_correct_answer(resp, ground_truth)
                reward = self.reward_fn(resp, ground_truth)
                num_tokens = rollout["response_mask"][0].sum().item()

                correct_count += is_corr
                total_reward += reward
                total_tokens += num_tokens

                if is_corr and not printed_correct:
                    print("\n[Eval] Correct Example Found:")
                    self._print_example(qs[j], resp)
                    printed_correct = True
                elif not is_corr and not printed_incorrect:
                    print(
                        f"\n[Eval] Incorrect Example Found (GT={ground_truth}):"
                    )
                    self._print_example(qs[j], resp)
                    printed_incorrect = True

        self.model.train()

        return {
            "eval/reward": total_reward / n_total,
            "eval/accuracy": correct_count / n_total,
            "eval/avg_response_length": total_tokens / n_total,
            "eval/samples": n_total,
        }

    def train(self, dataset, eval_dataset=None, run_name="grpo_model"):
        wandb.init(
            project="grpo-qwen3-gsm8k",
            name=run_name,
            mode="online" if self.config.use_wandb else "disabled",
            config={
                "model": "Qwen3-1.7B",
                "dataset": "gsm8k",
                "G": self.config.G,
                "batch_size": self.config.batch_size,
                "K": self.config.K,
                "lr": self.config.lr,
                "epsilon": self.config.epsilon,
                "temperature": self.config.temperature,
                "max_new_tokens": self.config.max_new_tokens,
                "max_steps": self.config.max_steps,
                "grad_clip": self.config.grad_clip,
            },
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        # Each batch produces K optimizer steps
        batches_per_epoch = len(dataloader)
        steps_per_epoch = batches_per_epoch * self.config.K
        max_epochs = math.ceil(self.config.max_steps / steps_per_epoch)

        print(
            f"Training: {self.config.max_steps} optimizer steps | "
            f"{batches_per_epoch} batches/epoch | "
            f"K={self.config.K} updates/batch | "
            f"{steps_per_epoch} steps/epoch | "
            f"~{max_epochs} epochs over {len(dataset)} questions"
        )
        print(
            f"Batch size: {self.config.batch_size} | "
            f"G: {self.config.G} | "
            f"Completions per batch: {self.config.batch_size * self.config.G}"
        )
        if eval_dataset:
            print(
                f"Evaluating every {self.config.eval_every} optimizer steps "
                f"on {self.config.eval_samples} samples"
            )
        print("-" * 50)

        # Initial evaluation
        if eval_dataset:
            metrics = self.evaluate(eval_dataset)
            best_accuracy = metrics["eval/accuracy"]
            self._log_eval(0, metrics)
        else:
            best_accuracy = 0.0

        self.model.train()
        global_step = 0
        examples_seen = 0

        for epoch in range(max_epochs):
            epoch_loss = 0.0
            epoch_batches = 0

            for batch in dataloader:
                if global_step >= self.config.max_steps:
                    break

                batch_q = batch["question"]
                batch_a = batch["answer"]
                examples_seen += len(batch_q)

                # 1. Generate rollouts
                self.model.eval()
                torch.cuda.synchronize()
                t0 = time.perf_counter()

                rollouts = generate_rollouts(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    questions=batch_q,
                    G=self.config.G,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                )

                torch.cuda.synchronize()
                gen_time = time.perf_counter() - t0
                self.model.train()

                gen_tokens = sum(
                    r["response_mask"].sum().item() for r in rollouts
                )
                self.timings["generation"] += gen_time
                self.timings["total_gen_tokens"] += gen_tokens

                # 2. Cache old log-probs
                cached_logps = [self._cache_logps(r) for r in rollouts]

                # 3. Rewards and advantages
                all_rewards = []
                all_advantages = []
                for rollout, answer in zip(rollouts, batch_a):
                    rewards = self._compute_rewards(
                        rollout["response_texts"], answer
                    )
                    advantages = compute_advantages(rewards)
                    all_rewards.append(rewards)
                    all_advantages.append(advantages)

                # 4. Training metrics
                train_metrics = self._compute_batch_metrics(
                    rollouts, all_rewards
                )

                # 5. K policy update steps
                train_time = 0.0
                for k in range(self.config.K):
                    if global_step >= self.config.max_steps:
                        break

                    torch.cuda.synchronize()
                    k_t0 = time.perf_counter()

                    self.optimizer.zero_grad()
                    batch_loss = 0.0
                    for rollout, old_logps, adv in zip(
                        rollouts, cached_logps, all_advantages
                    ):
                        loss_val = self._train_step(
                            rollout, old_logps, adv, n_rollouts=len(rollouts)
                        )
                        batch_loss += loss_val

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.config.grad_clip,
                    )
                    self.optimizer.step()

                    torch.cuda.synchronize()
                    train_time += time.perf_counter() - k_t0

                    global_step += 1

                    wandb.log(
                        {
                            **train_metrics,
                            "train/loss": batch_loss,
                            "train/examples_seen": examples_seen,
                            "train/k_epoch": k,
                        },
                        step=global_step,
                    )

                    # Eval after optimizer step
                    if (
                        eval_dataset
                        and global_step % self.config.eval_every == 0
                    ):
                        metrics = self.evaluate(eval_dataset)
                        self._log_eval(global_step, metrics)

                        if metrics["eval/accuracy"] > best_accuracy:
                            best_accuracy = metrics["eval/accuracy"]
                            save_model(
                                self.model,
                                self.tokenizer,
                                f"{run_name}-best",
                            )
                            print(
                                f"  New best model saved at step {global_step}"
                            )

                # Update timing stats
                train_tokens = gen_tokens * self.config.K
                self.timings["training"] += train_time
                self.timings["total_train_tokens"] += train_tokens

                epoch_loss += batch_loss
                epoch_batches += 1

                # Console + perf logging every 10 batches
                if epoch_batches % 10 == 0:
                    total_time = (
                        self.timings["generation"] + self.timings["training"]
                    )
                    gen_pct = (
                        100
                        * self.timings["generation"]
                        / max(total_time, 1e-9)
                    )
                    gen_tps = self.timings["total_gen_tokens"] / max(
                        self.timings["generation"], 1e-9
                    )
                    train_tps = self.timings["total_train_tokens"] / max(
                        self.timings["training"], 1e-9
                    )
                    examples_per_hour = examples_seen / max(
                        total_time / 3600, 1e-9
                    )

                    print(
                        f"  Step {global_step:>4d} | "
                        f"Examples: {examples_seen:>6d} | "
                        f"Reward: {train_metrics['train/mean_reward']:.3f} | "
                        f"Correct: {train_metrics['train/fraction_correct']:.1%} | "
                        f"Gen: {gen_tps:.0f} tok/s ({gen_pct:.0f}%) | "
                        f"Train: {train_tps:.0f} tok/s | "
                        f"{examples_per_hour:.0f} ex/hr"
                    )

                    wandb.log(
                        {
                            "perf/gen_tokens_per_sec": gen_tps,
                            "perf/train_tokens_per_sec": train_tps,
                            "perf/gen_time_pct": gen_pct,
                            "perf/examples_per_hour": examples_per_hour,
                            "perf/wall_time": total_time,
                        },
                        step=global_step,
                    )

            if global_step >= self.config.max_steps:
                break

            avg_epoch_loss = epoch_loss / max(epoch_batches, 1)
            self._log_epoch(epoch, avg_epoch_loss)

        # Final evaluation
        if eval_dataset:
            metrics = self.evaluate(eval_dataset)
            self._log_eval(global_step, metrics)

        wandb.finish()
        self.model.eval()
        print(
            f"Training complete. {global_step} optimizer steps, "
            f"{examples_seen} examples seen, ~{epoch + 1} epochs."
        )

    def _print_example(self, question, response):
        lmprint.print_question(question)
        lmprint.pretty_print(response)

    def _log_epoch(self, epoch, avg_loss):
        print(f"Epoch {epoch + 1:>2d} | Avg Loss: {avg_loss:.4f}")
        print("-" * 50)

    def _log_eval(self, step, metrics):
        print(
            f"\n  [EVAL @ step {step}] "
            f"Reward: {metrics['eval/reward']:.3f} | "
            f"Accuracy: {metrics['eval/accuracy']:.1%} | "
            f"Avg Len: {metrics['eval/avg_response_length']:.1f} tokens | "
            f"Samples: {metrics['eval/samples']}\n"
        )
        wandb.log(metrics, step=step)