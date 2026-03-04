import re
from collections import Counter

import torch
from datasets import load_dataset

from grpo.functions import generate_completions


def generate_prompt(
    question,
    helper="",
    think_start_tok="<think>",
    think_stop_tok="</think>",
    answer_start_tok="<answer>",
    answer_stop_tok="</answer>",
):
    """
    Wraps a question into the DeepSeek-R1 paper prompt format.
    """
    prompt = (
        "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
        "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
        f"The reasoning process and answer are enclosed within {think_start_tok}...{think_stop_tok} and {answer_start_tok}...{answer_stop_tok} tags, "
        f"respectively, i.e., {think_start_tok} reasoning process here {think_stop_tok} {answer_start_tok} answer here {answer_stop_tok}. "
        f"User: {question}. Assistant: {helper}"
    )
    return prompt


# ---------- parsing helpers ----------

def extract_gsm8k_gold(answer_str: str) -> int | None:
    # GSM8K answers contain: "\n#### 42"
    m = re.search(r"####\s*(-?\d+)", answer_str)
    return int(m.group(1)) if m else None


def extract_answer_tag_int(text: str) -> int | None:
    # Prefer <answer> ... </answer>
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    inner = m.group(1)
    nums = re.findall(r"-?\d+", inner)
    return int(nums[-1]) if nums else None


def extract_predicted_int(text: str) -> int | None:
    # 1) Prefer answer-tag extraction
    v = extract_answer_tag_int(text)
    if v is not None:
        return v

    # 2) Fallback: last integer anywhere
    nums = re.findall(r"-?\d+", text)
    return int(nums[-1]) if nums else None


def majority_vote(values: list[int | None]) -> int | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return Counter(vals).most_common(1)[0][0]


# ---------- evaluation ----------

@torch.no_grad()
def evaluate_gsm8k_maj16(
    model,
    tokenizer,
    split: str = "test",
    n_examples: int | None = None,
    batch_size: int = 8,
    G: int = 16,
    temperature: float = 0.7,
    max_new_tokens: int = 256,
    helper: str = "<think>",
    show_examples: int = 0,
):
    """
    maj@16 evaluation:
      - sample 16 completions per question
      - extract integer answer
      - majority vote
      - exact match vs GSM8K gold integer

    Returns:
      maj@16_accuracy, pass@16_accuracy, no_extract_rate
    """
    ds = load_dataset("gsm8k", "main")[split]

    if n_examples is not None:
        ds = ds.select(range(min(n_examples, len(ds))))

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    total = 0
    correct_maj = 0
    correct_pass = 0
    no_extract = 0
    example_printed = 0

    for start in range(0, len(ds), batch_size):
        batch = ds.select(range(start, min(start + batch_size, len(ds))))
        prompt_texts = [generate_prompt(ex["question"], helper=helper) for ex in batch]

        # Generate 16 sampled completions per prompt
        _, _, all_texts, all_group_idx = generate_completions(
            model=model,
            tokenizer=tokenizer,
            prompt_texts=prompt_texts,
            G=G,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        # Group the generated outputs by prompt index (within the batch)
        grouped = [[] for _ in range(len(prompt_texts))]
        for txt, gi in zip(all_texts, all_group_idx):
            grouped[gi].append(txt)

        # Score
        for i, ex in enumerate(batch):
            gold = extract_gsm8k_gold(ex["answer"])
            samples = grouped[i]  # 16 strings

            preds = [extract_predicted_int(t) for t in samples]
            maj_pred = majority_vote(preds)

            if example_printed < show_examples:
                print("\n--- Example ---")
                print("Q:", ex["question"])
                print("Gold:", gold)
                print("maj@16 pred:", maj_pred)
                print("Sample[0] extracted:", preds[0])
                print("Sample[0] text (truncated):", samples[0][:300].replace("\n", " "))
                example_printed += 1

            if maj_pred is None:
                no_extract += 1

            is_correct_maj = (maj_pred is not None and gold is not None and maj_pred == gold)
            is_correct_pass = any(p is not None and gold is not None and p == gold for p in preds)

            correct_maj += int(is_correct_maj)
            correct_pass += int(is_correct_pass)
            total += 1

    return {
        "maj@16_accuracy": correct_maj / total,
        "pass@16_accuracy": correct_pass / total,
        "no_extract_rate": no_extract / total,
        "total": total,
        "G": G,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
    }