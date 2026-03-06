import re
from collections import Counter

import math
import torch
from datasets import load_dataset

from grpo.functions import generate_completions


# ---------- Generators ---------- #

def generate_prompt_deepseek(
    question,
    helper="",
    think_start_tok="<think>",
    think_stop_tok="</think>",
    answer_start_tok="<answer>",
    answer_stop_tok="</answer>",
):
    # DeepSeek-R1 paper prompt format.
    prompt = (
        "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
        "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
        f"The reasoning process and answer are enclosed within {think_start_tok}...{think_stop_tok} and {answer_start_tok}...{answer_stop_tok} tags, "
        f"respectively, i.e., {think_start_tok} reasoning process here {think_stop_tok} {answer_start_tok} answer here {answer_stop_tok}. "
        f"User: {question}. Assistant: {helper}"
    )
    return prompt


def generate_prompt_base(
    question,
    helper="",
    think_start_tok="<think>",
    think_stop_tok="</think>",
    answer_start_tok="<answer>",
    answer_stop_tok="</answer>",
):
    # A minimal prompt that keeps the same tag structure
    prompt = (
        "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
        f"The answer is enclosed within {answer_start_tok}...{answer_stop_tok} tags, "
        f"i.e., {answer_start_tok} answer here {answer_stop_tok}. "
        f"User: {question}. Assistant: {helper}"
    )
    return prompt


def generate_prompt(
    question,
    helper="",
    prompt_style: str = "deepseek",
    think_start_tok="<think>",
    think_stop_tok="</think>",
    answer_start_tok="<answer>",
    answer_stop_tok="</answer>",
):
    # Choose prompt type
    style = (prompt_style or "deepseek").lower()
    if style in {"base", "minimal"}:
        return generate_prompt_base(
            question,
            helper=helper,
            think_start_tok=think_start_tok,
            think_stop_tok=think_stop_tok,
            answer_start_tok=answer_start_tok,
            answer_stop_tok=answer_stop_tok,
        )
    return generate_prompt_deepseek(
        question,
        helper=helper,
        think_start_tok=think_start_tok,
        think_stop_tok=think_stop_tok,
        answer_start_tok=answer_start_tok,
        answer_stop_tok=answer_stop_tok,
    )



# ---------- Parsing helpers ---------- #

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
    # Prefer <answer>...</answer>
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        inner = m.group(1)
    else:
        inner = text

    # Find last number that can be int/float, allowing $, commas, etc.
    nums = re.findall(r"-?\d+(?:\.\d+)?", inner.replace(",", ""))
    if not nums:
        return None

    x = float(nums[-1])

    # GSM8K answers are integers; accept example 18.0
    if abs(x - round(x)) < 1e-6:
        return int(round(x))

    # If it's not an integer, return None
    return None


def majority_vote(values: list[int | None]) -> int | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return Counter(vals).most_common(1)[0][0]


def count_think_tokens(text: str, tokenizer) -> int:
    """Count tokenizer tokens inside <think>...</think>. Returns 0 if missing."""
    m = re.search(r"<think>\s*(.*?)\s*</think>", text, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return 0
    think_text = m.group(1)
    # Tokenize without special tokens for a fair token count
    return len(tokenizer.encode(think_text, add_special_tokens=False))


def count_total_tokens(text: str, tokenizer) -> int:
    """Count total tokenizer tokens in the generated completion string."""
    return len(tokenizer.encode(text, add_special_tokens=False))


# ---------- Evaluation ---------- #

@torch.no_grad()
def evaluate_gsm8k_maj16(
    model,
    tokenizer,
    split: str = "test",
    n_examples: int | None = None,
    batch_size: int = 8,
    maj_k: int = 16,        # Choose which maj@ to use
    pass_k: int = 1,        # CHoose which pass@ to use
    temperature: float = 0.7,
    max_new_tokens: int = 256,
    helper: str = "<think>",
    prompt_style: str = "deepseek",
    show_examples: int = 0,
):
    """
    maj@maj_k evaluation:
      - sample max(maj_k, pass_k) completions per question
      - extract integer answer
      - majority vote
      - exact match vs GSM8K gold integer

    Returns:
      maj@maj_k_accuracy, pass@pass_k_accuracy, no_extract_rate
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
    total_think_tokens = 0
    total_completions = 0
    total_completion_tokens = 0

    for start in range(0, len(ds), batch_size):
        batch = ds.select(range(start, min(start + batch_size, len(ds))))
        prompt_texts = [
            generate_prompt(ex["question"], helper=helper, prompt_style=prompt_style)
            for ex in batch
        ]

        # Generate max(maj_k, pass_k) sampled completions per prompt
        _, _, all_texts, all_group_idx = generate_completions(
            model=model,
            tokenizer=tokenizer,
            prompt_texts=prompt_texts,
            G=max(maj_k, pass_k),
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

            samples_all = grouped[i]
            samples_maj = samples_all[:maj_k]
            samples_pass = samples_all[:pass_k]

            completion_counts_maj = [count_total_tokens(s, tokenizer) for s in samples_maj]
            total_completion_tokens += sum(completion_counts_maj)

            think_counts_maj = [count_think_tokens(s, tokenizer) for s in samples_maj]
            total_think_tokens += sum(think_counts_maj)
            total_completions += len(think_counts_maj)

            preds_all = [extract_predicted_int(t) for t in samples_all]
            preds_maj = preds_all[:maj_k]
            preds_pass = preds_all[:pass_k]

            maj_pred = majority_vote(preds_maj)

            if example_printed < show_examples:
                print("\n--- Example ---")
                print("Q:", ex["question"])
                print("Gold (correct answer):", gold)
                print(f"maj@{maj_k} pred:", maj_pred)

                valid_votes = [p for p in preds_maj if p is not None]
                print("\nValid votes:", len(valid_votes), "of", len(preds_maj))
                print("Top votes:", Counter(valid_votes).most_common(5))

                # Thinking token usage (based on <think>...</think>)
                sample0_think = count_think_tokens(samples_all[0], tokenizer)
                avg_think_maj = (sum(think_counts_maj) / len(think_counts_maj)) if think_counts_maj else 0.0
                print("Think tokens:", sample0_think)
                print(f"Avg think tokens:", round(avg_think_maj, 2))

                sample0_total = count_total_tokens(samples_all[0], tokenizer)
                avg_total_maj = (sum(completion_counts_maj) / len(completion_counts_maj)) if completion_counts_maj else 0.0
                print("Completion tokens:", sample0_total)
                print(f"Avg completion tokens:", round(avg_total_maj, 2))

                example_printed += 1

            if maj_pred is None:
                no_extract += 1

            is_correct_maj = (maj_pred is not None and gold is not None and maj_pred == gold)
            is_correct_pass = any(p is not None and gold is not None and p == gold for p in preds_pass)

            correct_maj += int(is_correct_maj)
            correct_pass += int(is_correct_pass)
            total += 1

    return {
        f"maj@{maj_k}_accuracy": correct_maj / total,
        f"pass@{pass_k}_accuracy": correct_pass / total,
        "no_extract_rate": no_extract / total,
        "total": total,
        "maj_k": maj_k,
        "pass_k": pass_k,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "prompt_style": prompt_style,
        "avg_think_tokens_per_maj_sample": (total_think_tokens / total_completions) if total_completions else 0.0,
        "avg_completion_tokens_per_maj_sample": (total_completion_tokens / total_completions) if total_completions else 0.0,
    }