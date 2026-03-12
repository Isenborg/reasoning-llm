import re
from collections import Counter

import math
import torch
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import load_dataset
from torch.utils.data import DataLoader
from utils.models import DEFAULT_MODEL_DIR


# ---------- Generators ---------- #
def generate_completions(prompt, model, tokenizer, temperature, num_responses, max_new_tokens): 
    """
    Returns:
        * Response text
        * Num tokens in responses
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device) 
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.95,
            do_sample=True,
            num_return_sequences=num_responses,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response_toks = outputs[:, input_len:]
    response_texts = tokenizer.batch_decode(response_toks, skip_special_tokens=True)
    
    token_counts = [
        (row != tokenizer.pad_token_id).sum().item()
        for row in response_toks
    ]
    return response_texts, token_counts


def generate_prompt_deepseek(
    question,
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
        f"User: {question}. Assistant: "
    )
    return prompt


def generate_prompt_base(
    question,
    think_start_tok="<think>",
    think_stop_tok="</think>",
    answer_start_tok="<answer>",
    answer_stop_tok="</answer>",
):
    # A minimal prompt that keeps the same tag structure
    prompt = (
        "A conversation between User and Assistant. The user asks a math question, and the Assistant solves it. "
        f"The answer is enclosed within {answer_start_tok}...{answer_stop_tok} tags, "
        f"i.e., {answer_start_tok} answer here {answer_stop_tok}. "
        f"User: {question}. Assistant: "
    )
    return prompt

def generate_prompt(
    question,
    prompt_style: str = "reasoning",
    think_start_tok="<think>",
    think_stop_tok="</think>",
    answer_start_tok="<answer>",
    answer_stop_tok="</answer>",
):
    # Choose prompt type
    style = prompt_style.lower()
    if style in {"base", "minimal"}:
        return generate_prompt_base(
            question,
            answer_start_tok=answer_start_tok,
            answer_stop_tok=answer_stop_tok,
        )
    return generate_prompt_deepseek(
        question,
        think_start_tok=think_start_tok,
        think_stop_tok=think_stop_tok,
        answer_start_tok=answer_start_tok,
        answer_stop_tok=answer_stop_tok,
    )



# ---------- Parsing helpers ---------- #
# The parsing of the answer can be more loose than for the actual RL training, where the 
# answer HAS to be within the answer tags. 
# While here we will evaluate both reasoning and non reasoning models, we can basically check instead
# if the responses final answer was correct, not caring about formating as much.

def extract_predicted_int(text: str) -> int | None:
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        inner = m.group(1)
    else:
        inner = text

    nums = re.findall(r"-?\d+(?:\.\d+)?", inner.replace(",", ""))
    if not nums:
        return None

    x = float(nums[-1])

    if math.isinf(x):
        print("Model collapse, generated ", x)
        return None

    if abs(x - round(x)) < 1e-6:
        return int(round(x))
    return None



def majority_vote(values: list[int | None]) -> int | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return Counter(vals).most_common(1)[0][0]


# ---------- Evaluation ---------- #
@torch.no_grad()
def evaluate_gsm8k_sequential(
    model,
    tokenizer,
    dataset,
    n_examples: int | None = None,
    maj_k: int = 16,
    temperature: float = 0.7,
    max_new_tokens: int = 1024,
    prompt_style: str = "reasoning",
):
    n_total = n_examples if n_examples is not None else len(dataset)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    maj_correct_count = 0
    pass_k_result = 0.0
    average_response_lengths = []

    pbar = tqdm(loader, total=n_total, desc="Evaluating")

    for step, item in enumerate(pbar):
        if step >= n_total:
            break

        step_start = time.perf_counter()

        question = item["question"][0]
        answer = int(item["answer"][0])
        prompt_text = generate_prompt(question, prompt_style=prompt_style)

        response_texts, response_lengths = generate_completions(
            prompt_text,
            model,
            tokenizer,
            temperature,
            maj_k,          # ← you were missing a comma here
            max_new_tokens,
        )

        preds = [extract_predicted_int(response) for response in response_texts]

        n_correct = sum(p == answer for p in preds)  # ← was preds == answer
        pass_k_result += n_correct / maj_k
        maj_correct_count += (majority_vote(preds) == answer)
        average_response_lengths.append(sum(response_lengths) / maj_k)

        step_time = time.perf_counter() - step_start

        # Running averages
        n = step + 1
        pbar.set_postfix({
            "maj@k":    f"{maj_correct_count / n:.3f}",
            "pass@1":   f"{pass_k_result / n:.3f}",
            "avg_toks":  f"{sum(average_response_lengths) / n:.0f}",
            "step_time": f"{step_time:.1f}s",
        })

    pbar.close()

    n = min(n_total, len(dataset))
    print(f"\nFinal Results ({n} examples):")
    print(f"  maj@{maj_k}:  {maj_correct_count / n:.4f}")
    print(f"  pass@1:   {pass_k_result / n:.4f}")
    print(f"  avg tokens/response: {sum(average_response_lengths) / n:.0f}")


# We can use vllm for the generation, should be a lot faster
from vllm import LLM, SamplingParams
import random
import os
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"
os.environ["VLLM_CONFIGURE_LOGGING"] = "0"


@torch.no_grad()
def evaluate_gsm8k_vllm(
    model_name: str,
    dataset,
    n_examples: int = -1,
    maj_k: int = 16,
    temperature: float = 0.7,
    max_new_tokens: int = 1024,
    prompt_style: str = "reasoning",
):
    n_total = n_examples if n_examples is not -1 else len(dataset)

    # ─── Prepare ALL prompts upfront ───
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    indices = indices[:n_total]

    prompts = []
    answers = []
    for idx in indices:
        item = dataset[idx]
        prompts.append(generate_prompt(item["question"], prompt_style=prompt_style))
        answers.append(int(item["answer"]))

    # ─── Load model ───
    model_path = DEFAULT_MODEL_DIR / model_name
    llm = LLM(
        model=str(model_path),
        dtype="float16",
        gpu_memory_utilization=0.90,
        max_model_len=2048,
    )

    # ─── Greedy pass@1 ───
    print(f"Generating {n_total} greedy completions...")
    greedy_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
        n=1,
    )
    greedy_start = time.perf_counter()
    greedy_outputs = llm.generate(prompts, greedy_params)
    greedy_time = time.perf_counter() - greedy_start
    print(f"Greedy generation done in {greedy_time:.1f}s")

    # ─── Sampled maj@k ───
    print(f"Generating {n_total} × {maj_k} = {n_total * maj_k} sampled completions...")
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_new_tokens,
        n=maj_k,
    )
    sampled_start = time.perf_counter()
    sampled_outputs = llm.generate(prompts, sampling_params)
    sampled_time = time.perf_counter() - sampled_start
    print(f"Sampled generation done in {sampled_time:.1f}s")

    # ─── Score greedy ───
    greedy_correct = 0
    for i, output in enumerate(greedy_outputs):
        pred = extract_predicted_int(output.outputs[0].text)
        greedy_correct += (pred == answers[i])

    # ─── Score sampled ───
    maj_correct = 0
    sampled_pass_1_sum = 0.0
    avg_lengths = []
    failed_extractions = 0
    total_responses = 0

    for i, output in enumerate(tqdm(sampled_outputs, desc="Scoring")):
        answer = answers[i]
        response_texts = [o.text for o in output.outputs]
        response_lengths = [len(o.token_ids) for o in output.outputs]

        preds = [extract_predicted_int(r) for r in response_texts]

        n_correct = sum(p == answer for p in preds)
        n_failed = sum(p is None for p in preds)

        sampled_pass_1_sum += n_correct / maj_k
        maj_correct += (majority_vote(preds) == answer)
        avg_lengths.append(sum(response_lengths) / maj_k)
        failed_extractions += n_failed
        total_responses += maj_k

    # ─── Report ───
    n = len(sampled_outputs)
    total_tokens = sum(sum(len(o.token_ids) for o in out.outputs) for out in sampled_outputs)
    total_time = greedy_time + sampled_time

    print(f"\nResults ({n} examples):")
    print(f"  Model used:          {model_name}")
    print(f"  pass@1 (greedy):     {greedy_correct / n:.4f}")
    print(f"  pass@1 (sampled):    {sampled_pass_1_sum / n:.4f}")
    print(f"  maj@{maj_k}:              {maj_correct / n:.4f}")
    print(f"  degeneration rate:   {failed_extractions / total_responses:.4f}")
    print(f"  temperature:         {temperature}")
    print(f"  avg tok/resp:        {sum(avg_lengths) / n:.0f}")
    print(f"  throughput:          {total_tokens / sampled_time:.0f} tok/s")
    print(f"  total time:          {total_time:.1f}s")


# ---------- SFT evaluation with plots ---------- #
from grpo.utils import generate_rollouts



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

def count_think_tokens(text: str, tokenizer) -> int:
    """Count tokenizer tokens inside <think>...</think>. Returns 0 if missing."""
    m = re.search(r"<think>\s*(.*?)\s*</think>", text, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return 0
    think_text = m.group(1)
    # Tokenize without special tokens for a fair token count
    return len(tokenizer.encode(think_text, add_special_tokens=False))

@torch.no_grad()
def evaluate_sft(
    model,
    tokenizer,
    split: str = "test",
    n_examples: int | None = None,
    batch_size: int = 8,
    temperature: float = 0.3,
    max_new_tokens: int = 512,
    helper: str = "<think>",
    prompt_style: str = "deepseek",
    show_examples: int = 2,
):
    """
    Evaluate the SFT model on GSM8K and produce two plots:
      1. Cumulative accuracy over questions
      2. Distribution of think tokens per response

    Returns a metrics dict and the per-question records.
    """
    ds = load_dataset("gsm8k", "main")[split]
    if n_examples is not None:
        ds = ds.select(range(min(n_examples, len(ds))))

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    records = []          # one dict per question
    example_printed = 0

    for start in range(0, len(ds), batch_size):
        batch = ds.select(range(start, min(start + batch_size, len(ds))))
        questions = [
            generate_prompt(ex["question"], helper=helper, prompt_style=prompt_style)
            for ex in batch
        ]

        # Generate one completion per prompt
        rollouts = generate_rollouts(
            model=model,
            tokenizer=tokenizer,
            questions=questions,
            G=1,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        for i, ex in enumerate(batch):
            completion = rollouts[i]["response_texts"][0] if rollouts[i]["response_texts"] else ""
            gold = extract_gsm8k_gold(ex["answer"])
            

            pred = extract_predicted_int(completion)
            correct = (pred is not None and gold is not None and pred == gold)
            think_toks = count_think_tokens(completion, tokenizer)
            has_think = think_toks > 0
            has_answer = extract_answer_tag_int(completion) is not None

            records.append({
                "question": ex["question"],
                "gold": gold,
                "pred": pred,
                "correct": correct,
                "think_tokens": think_toks,
                "has_think": has_think,
                "has_answer": has_answer,
                "completion": completion,
            })

            if example_printed < show_examples:
                print(f"\n--- Example {example_printed + 1} ---")
                print(f"Q: {ex['question'][:120]}...")
                print(f"Gold: {gold}  |  Pred: {pred}  |  {'CORRECT' if correct else 'WRONG'}")
                print(f"Think tokens: {think_toks}")
                print(f"Response: {completion[:300]}...")
                example_printed += 1

    # ── Metrics ──────────────────────────────────────────────────────────────
    total = len(records)
    n_correct = sum(r["correct"] for r in records)
    n_no_extract = sum(r["pred"] is None for r in records)
    n_has_think = sum(r["has_think"] for r in records)
    think_tokens = [r["think_tokens"] for r in records]
    avg_think = sum(think_tokens) / total if total else 0.0

    metrics = {
        "accuracy": n_correct / total,
        "no_extract_rate": n_no_extract / total,
        "think_block_rate": n_has_think / total,
        "avg_think_tokens": avg_think,
        "total": total,
    }

    print(f"\n{'='*50}")
    print(f"GSM8K {split} — {total} examples")
    print(f"  Accuracy        : {metrics['accuracy']:.1%}")
    print(f"  No extract rate : {metrics['no_extract_rate']:.1%}")
    print(f"  Think block rate: {metrics['think_block_rate']:.1%}")
    print(f"  Avg think tokens: {avg_think:.1f}")
    print(f"{'='*50}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    _plot_sft_results(records, metrics)

    return metrics, records


def _plot_sft_results(records, metrics):
    """Two-panel plot: cumulative accuracy + think token distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    # --- Panel 1: Cumulative accuracy ---
    cumulative_correct = 0
    cumulative_acc = []
    for r in records:
        cumulative_correct += r["correct"]
        cumulative_acc.append(cumulative_correct / (len(cumulative_acc) + 1))

    ax1.plot(range(1, len(cumulative_acc) + 1), cumulative_acc,
             color="steelblue", linewidth=1.5)
    ax1.axhline(metrics["accuracy"], color="tomato", linestyle="--", linewidth=1.2,
                label=f"Final accuracy: {metrics['accuracy']:.1%}")
    ax1.set_xlabel("Questions evaluated")
    ax1.set_ylabel("Cumulative accuracy")
    ax1.set_title("Accuracy on GSM8K test set")
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Think token distribution ---
    think_tokens = [r["think_tokens"] for r in records]
    n_zero = sum(t == 0 for t in think_tokens)
    n_nonzero = [t for t in think_tokens if t > 0]

    if n_nonzero:
        ax2.hist(n_nonzero, bins=30, color="mediumpurple", alpha=0.8, edgecolor="white")
    ax2.axvline(metrics["avg_think_tokens"], color="tomato", linestyle="--", linewidth=1.5,
                label=f"Mean: {metrics['avg_think_tokens']:.0f} tokens")
    ax2.set_xlabel("Think tokens per response")
    ax2.set_ylabel("Number of responses")
    ax2.set_title(f"Think token distribution\n({n_zero}/{len(records)} responses had no <think> block)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle("SFT Model — GSM8K Evaluation", fontweight="bold")
    plt.tight_layout()
    plt.show()