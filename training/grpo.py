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
            ids = rollout["input_ids"][i:i+1]
            mask = rollout["attention_mask"][i:i+1]
            r_mask = rollout["response_mask"][i:i+1]
            old_lp = old_logps[i:i+1]
            adv = advantages[i:i+1]

            new_lp = get_per_token_logps(self.model, ids, mask)
            log_ratio = (new_lp - old_lp) * r_mask
            ratios = torch.exp(log_ratio)

            loss = grpo_loss(ratios, adv, r_mask, epsilon=self.config.epsilon)

            # divide by accumulation steps so gradient scale is correct
            scaled = loss / (G * n_rollouts * self.config.gradient_accumulation_steps)
            scaled.backward()

            total_loss += loss.item()

        return total_loss / G


    @torch.no_grad()
    def evaluate(self, eval_dataset):
        self.model.eval()

        # 1. Setup total count and batch size
        n_total = min(self.config.eval_samples, len(eval_dataset))
        indices = random.sample(range(len(eval_dataset)), n_total)
        batch_size = 32
        
        total_reward = 0.0
        correct_count = 0
        total_tokens = 0.0
        
        printed_correct = False
        printed_incorrect = False

        # 2. Loop through the requested samples in batches of 32
        for i in range(0, n_total, batch_size):
            batch_indices = indices[i : i + batch_size]
            batch_examples = [eval_dataset[idx] for idx in batch_indices]
            
            qs = [ex["question"] for ex in batch_examples]
            ans = [ex["answer"] for ex in batch_examples]

            # Generate for the whole batch (much faster)
            rollouts = generate_rollouts(
                model=self.model,
                tokenizer=self.tokenizer,
                questions=qs,
                G=1,
                max_new_tokens=self.config.max_new_tokens,
                temperature=0.3,
            )

            # 3. Accumulate results from this batch
            for j, rollout in enumerate(rollouts):
                resp = rollout["response_texts"][0]
                ground_truth = ans[j]
                
                # Metrics
                is_corr = is_correct_answer(resp, ground_truth)
                reward = self.reward_fn(resp, ground_truth)
                num_tokens = rollout["response_mask"][0].sum().item()

                correct_count += is_corr
                total_reward += reward
                total_tokens += num_tokens

                # 4. Simple Example Printing (one of each per eval run)
                if is_corr and not printed_correct:
                    print("\n[Eval] Correct Example Found:")
                    self._print_example(qs[j], resp)
                    printed_correct = True
                elif not is_corr and not printed_incorrect:
                    print(f"\n[Eval] Incorrect Example Found (GT={ground_truth}):")
                    self._print_example(qs[j], resp)
                    printed_incorrect = True

        self.model.train()
        
        return {
            "eval_reward": total_reward / n_total,
            "eval_accuracy": correct_count / n_total,
            "eval_avg_response_length": total_tokens / n_total,
            "eval_samples": n_total,
        }

    def train(self, dataset, eval_dataset=None, run_name="grpo_model"):
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        steps_per_epoch = len(dataloader)
        max_epochs = math.ceil(self.config.max_steps / steps_per_epoch)
        best_accuracy = 0.0

        # NEW: effective batch size for logging
        effective_batch = self.config.batch_size * self.config.gradient_accumulation_steps
        print(f"Training: {self.config.max_steps} steps | "
            f"{steps_per_epoch} steps/epoch | "
            f"~{max_epochs} epochs over {len(dataset)} questions")
        print(f"Batch size: {self.config.batch_size} x "
            f"{self.config.gradient_accumulation_steps} accum = "
            f"{effective_batch} effective")

        if eval_dataset:
            metrics = self.evaluate(eval_dataset)
            best_accuracy = metrics["eval_accuracy"]
            self._log_eval(0, metrics)

        self.model.train()
        global_step = 0
        accum_step = 0

        # zero gradients once at the start
        self.optimizer.zero_grad()

        for epoch in range(max_epochs):
            epoch_loss = 0.0
            num_steps = 0

            for batch in dataloader:
                if global_step >= self.config.max_steps:
                    break

                batch_q = batch["question"]
                batch_a = batch["answer"]

                # 1. Generate rollouts
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

                # 2. Cache old log-probs
                cached_logps = [self._cache_logps(r) for r in rollouts]

                # 3. Compute rewards and advantages
                all_rewards = []
                all_advantages = []
                for rollout, answer in zip(rollouts, batch_a):
                    rewards = self._compute_rewards(rollout["response_texts"], answer)
                    advantages = compute_advantages(rewards)
                    all_rewards.append(rewards)
                    all_advantages.append(advantages)

                # 4. Compute gradients
                batch_loss = 0.0
                for rollout, old_logps, adv in zip(rollouts, cached_logps, all_advantages):
                    loss_val = self._train_step(rollout, old_logps, adv, n_rollouts=len(rollouts))
                    batch_loss += loss_val

                accum_step += 1

                # only step when we've accumulated enough
                if accum_step % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.config.grad_clip
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                epoch_loss += batch_loss
                num_steps += 1
                global_step += 1

                mean_reward = torch.cat(all_rewards).mean().item()
                self._log_step(global_step, batch_loss, mean_reward)

                if eval_dataset and global_step % self.config.eval_every == 0:
                    metrics = self.evaluate(eval_dataset)
                    self._log_eval(global_step, metrics)
                    if metrics["eval_accuracy"] > best_accuracy:
                        best_accuracy = metrics["eval_accuracy"]
                        save_model(self.model, self.tokenizer, f"{run_name}-best")

            if global_step >= self.config.max_steps:
                break
            self._log_epoch(epoch, epoch_loss / max(num_steps, 1))

        # use any remaining accumulated gradients
        if accum_step % self.config.gradient_accumulation_steps != 0:        
            torch.nn.utils.clip_grad_norm_(                                 
                self.model.parameters(), max_norm=self.config.grad_clip      
            )                                                                
            self.optimizer.step()                                            
            self.optimizer.zero_grad()                                       

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