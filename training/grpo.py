import torch
import math
import random
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer
from utils.checks import is_correct_answer
from utils import lmprint, extracts

from grpo.functions import (
    get_per_token_logps,
    compute_advantages,
    compute_kl_penalty,
    grpo_loss,
    grpo_loss_with_kl,
)
from grpo.utils import generate_rollouts
from training.configs import GRPOConfig
from utils.models import save_model


class GRPOTrainer:
    def __init__(
        self,
        model: PreTrainedModel,
        ref_model: PreTrainedModel | None,
        tokenizer: PreTrainedTokenizer,
        reward_fn,
        config: GRPOConfig,
    ):
        self.model = model
        self.model.gradient_checkpointing_enable()
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.config = config

        if self.ref_model is not None:
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False

        self.optimizer = AdamW(self.model.parameters(), lr=self.config.lr)

    def _compute_rewards(self, response_texts: list[str], ground_truth: str) -> torch.Tensor:
        rewards = [self.reward_fn(resp, ground_truth) for resp in response_texts]
        return torch.tensor(rewards, dtype=torch.float, device=self.model.device)

    def _cache_logps(self, rollout):
        with torch.no_grad():
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
            ids = rollout["input_ids"][i:i+1]             # [1, seq_len]
            mask = rollout["attention_mask"][i:i+1]
            r_mask = rollout["response_mask"][i:i+1]
            old_lp = old_logps[i:i+1]
            adv = advantages[i:i+1]

            new_lp = get_per_token_logps(self.model, ids, mask)
            log_ratio = (new_lp - old_lp) * r_mask
            ratios = torch.exp(log_ratio)

            loss = grpo_loss(ratios, adv, r_mask, epsilon=self.config.epsilon)
            (loss / (G * n_rollouts)).backward()  # scale by total sequences
            total_loss += loss.item()

        return total_loss / G


    @torch.no_grad()
    def evaluate(self, eval_dataset):
        """
        Evaluate model on a subset of the eval dataset.
        Returns dict with metrics.
        """
        self.model.eval()

        # Sample a subset
        n = min(self.config.eval_samples, len(eval_dataset))
        indices = random.sample(range(len(eval_dataset)), n)

        total_reward = 0.0
        correct = 0
        total = 0
        total_response_tokens = 0.0

        print_correct = True
        print_incorrect = True

        for idx in indices:
            example = eval_dataset[idx]
            question = example["question"]
            answer = example["answer"]

            rollouts = generate_rollouts(
                model=self.model,
                tokenizer=self.tokenizer,
                questions=[question],
                G=1,  # Single greedy-ish sample for eval
                max_new_tokens=self.config.max_new_tokens,
                temperature=0.3,  # Low temperature for eval
            )

            response_text = rollouts[0]["response_texts"][0]

            # Sum the response mask to get the number of generated tokens
            response_num_tokens = rollouts[0]["response_mask"][0].sum().item()
            total_response_tokens += response_num_tokens 

            # Print out the response and question once per eval for a wrong and correct answer:
            if is_correct_answer(response_text, answer) and print_correct:
                print("Correct example:")
                self._print_example(question, response_text)
                print_correct = False
            elif not is_correct_answer(response_text, answer) and print_incorrect:
                print("Incorrect example:")
                self._print_example(question, response_text)
                print_incorrect = False

            correct += is_correct_answer(response_text, answer)
            reward = self.reward_fn(response_text, answer)
            total_reward += reward
            total += 1.0 # Max reward per response
 

        metrics = {
            "eval_reward": total_reward / total,
            "eval_accuracy": correct / total,
            "eval_avg_num_tokens": total_response_tokens / total,
            "eval_samples": total,
        }

        self.model.train()
        return metrics

    def train(self, dataset, eval_dataset=None, run_name="grpo_model"):
        """
        Main GRPO training loop.

        Args:
            dataset:      Training dataset of (question, answer) pairs
            eval_dataset: Optional evaluation dataset
        """
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        # Count one epoch as going through all training examples
        steps_per_epoch = len(dataloader)
        max_epochs = math.ceil(self.config.max_steps / steps_per_epoch)

        print(f"Training: {self.config.max_steps} steps | "
              f"{steps_per_epoch} steps/epoch | "
              f"~{max_epochs} epochs over {len(dataset)} questions")
        
        if eval_dataset:
            print(f"Evaluating every {self.config.eval_every} steps "
                  f"on {self.config.eval_samples} samples")
        print("-" * 50)

        # Initial evaluation
        if eval_dataset:
            metrics = self.evaluate(eval_dataset)
            self._log_eval(0, metrics)

        self.model.train()
        global_step = 0

        for epoch in range(max_epochs):
            epoch_loss = 0.0
            num_steps = 0

            for batch in dataloader:
                if global_step >= self.config.max_steps:
                    break

                batch_q = batch["question"]
                batch_a = batch["answer"]

                # 1. Generate rollouts with current model
                self.model.eval()
                rollouts = generate_rollouts(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    questions=batch_q,
                    G=self.config.G,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                )
                self.model.train()

                # 2. Cache log-probs (π_θ_old)
                cached_logps = [self._cache_logps(r) for r in rollouts]

                # 3. Score and compute advantages
                all_rewards = []
                all_advantages = []
                for rollout, answer in zip(rollouts, batch_a):
                    rewards = self._compute_rewards(rollout["response_texts"], answer)
                    advantages = compute_advantages(rewards)
                    all_rewards.append(rewards)
                    all_advantages.append(advantages)

                    
                # 4. K gradient steps on same rollouts
                for k in range(self.config.K):
                    if global_step >= self.config.max_steps:
                        break

                    self.optimizer.zero_grad()

                    batch_loss = 0.0
                    for rollout, old_logps, adv in zip(rollouts, cached_logps, all_advantages):
                        loss_val = self._train_step(rollout, old_logps, adv, n_rollouts=len(rollouts))
                        batch_loss += loss_val

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                    self.optimizer.step()

                    avg_batch_loss = batch_loss
                    epoch_loss += avg_batch_loss
                    num_steps += 1
                    global_step += 1

                    mean_reward = torch.cat(all_rewards).mean().item()
                    self._log_step(global_step, avg_batch_loss, mean_reward)

                    # Periodic evaluation
                    if eval_dataset and global_step % self.config.eval_every == 0:
                        metrics = self.evaluate(eval_dataset)
                        self._log_eval(global_step, metrics)
                        
                        # Save best model
                        if metrics["eval_accuracy"] > best_accuracy:
                            best_accuracy = metrics["eval_accuracy"]
                            save_model(self.model, self.tokenizer, f"{run_name}-best")
                            print(f"  New best model saved at step {global_step}")

            if global_step >= self.config.max_steps:
                break

            avg_epoch_loss = epoch_loss / max(num_steps, 1)
            self._log_epoch(epoch, avg_epoch_loss)

        # Final evaluation
        if eval_dataset:
            metrics = self.evaluate(eval_dataset)
            self._log_eval(global_step, metrics)

        self.model.eval()
        print(f"Training complete. {global_step} steps, ~{epoch + 1} epochs.")

    def _print_example(self, question, response):
        lmprint.print_question(question)
        lmprint.pretty_print(response)

    def _log_step(self, step, loss, mean_reward):
        print(f"  Step {step:>4d} | Loss: {loss:.4f} | Mean Reward: {mean_reward:.3f}")

    def _log_epoch(self, epoch, avg_loss):
        print(f"Epoch {epoch + 1:>2d} | Avg Loss: {avg_loss:.4f}")
        print("-" * 50)

    def _log_eval(self, step, metrics):
        print(f"\n  [EVAL @ step {step}] "
              f"Reward: {metrics['eval_reward']:.3f} | "
              f"Accuracy: {metrics['eval_accuracy']:.1%} | "
              f"Avg Len: {metrics['eval_avg_response_length']:.1f} tokens | "
              f"Samples: {metrics['eval_samples']}\n")