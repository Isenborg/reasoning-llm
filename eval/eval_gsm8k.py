import re
from collections import Counter

import math
import torch
import time
from tqdm import tqdm
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