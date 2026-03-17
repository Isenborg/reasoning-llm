import re
import torch
from torch.utils.data import Dataset
from datasets import load_dataset


class GSM8KDataset(Dataset):
    """
    GSM8K dataset. Extracts final numeric answer from solution string.

    Raw answer format: "...#### 42"
    We extract: "42"
    """

    def __init__(self, split: str = "train"):
        raw = load_dataset("openai/gsm8k", "main", split=split)

        self.questions = []
        self.answers = []

        for example in raw:
            question = example["question"]
            answer = self._extract_answer(example["answer"])

            if answer is not None:
                self.questions.append(question)
                self.answers.append(answer)

        print(f"Loaded {len(self)} GSM8K examples ({split})")

    def _extract_answer(self, solution: str) -> str | None:
        """Extract final answer after #### marker."""
        match = re.search(r"####\s*(.+)", solution)
        if match:
            # Remove commas from numbers like "1,234"
            return match.group(1).strip().replace(",", "")
        return None

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return {
            "question": self.questions[idx],
            "answer": self.answers[idx],
        }


class GSM8KSFTDataset(Dataset):
    """
    GSM8K dataset formatted for supervised fine-tuning.

    Each example is tokenized into:
      prompt  → the question wrapped in the DeepSeek-R1 prompt template
      completion → <think>{chain_of_thought}</think><answer>{answer}</answer>

    Labels are -100 for prompt tokens so the loss is only computed on the
    completion (chain-of-thought + answer).
    """

    def __init__(self, tokenizer, prompt_template, split: str = "train", max_length: int = 1024):
        raw = load_dataset("multi-domain-reasoning/gsm8k", split=split)

        self.samples = []

        for example in raw:
            question = example["question"]
            solution = example["answer"]
            
            # Extract ground truth
            answer = self._extract_answer(solution)
            # Extract reasoning trace
            cot = example["reasoning_nemotron_70B"]
            # Nemotron reasoning uses "<reasoning> instead of <think> tokens", lets format the data for our model.
            reasoning = self._extract_reasoning(cot)

            if not answer:
                print(f"[SKIP] No answer extracted for solution: {solution[:100]!r}...")
                continue
            if not reasoning:
                print(f"[SKIP] No reasoning extracted for cot: {cot!r}...")
                continue

            prompt = prompt_template(question)
            completion = f"<think>\n{reasoning}\n</think>\n<answer>{answer}</answer>"

            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            completion_ids = tokenizer.encode(completion, add_special_tokens=False)
            completion_ids = completion_ids + [tokenizer.eos_token_id]  # ← add EOS

            full_ids = prompt_ids + completion_ids
            labels = [-100] * len(prompt_ids) + completion_ids

            if len(full_ids) > max_length:
                continue

            self.samples.append({
                "input_ids": torch.tensor(full_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
                "input_text": prompt,
                "completion_text": completion,
            })

        print(f"GSM8KSFTDataset: {len(self.samples)} examples ({split}, max_length={max_length})")

    def _extract_answer(self, solution: str) -> str | None:
        """Extract final answer after #### marker."""
        match = re.search(r"####\s*(.+)", solution)
        if match:
            # Remove commas from numbers like "1,234"
            return match.group(1).strip().replace(",", "")
        return None
    
    def _extract_reasoning(self, cot: str) -> str:
        """Extract reasoning from within <reasoning> </reasoning> tokens"""
        pattern = r"<reasoning>(.*?)</reasoning>"
        match = re.search(pattern, cot, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def longest_sequence(self) -> int:
        """Longest full sequence — needed for memory/max_length planning."""
        return max(sample["input_ids"].size(0) for sample in self.samples)
    
    def total_tokens(self) -> int:
        return sum(sample["input_ids"].size(0) for sample in self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]