import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from training.configs import GRPOConfig


def generate_prompt(question, helper=""):
    """
    Wraps a question into the DeepSeek-R1 prompt format.
    """
    prompt = (
        "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
        "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
        "The reasoning process and answer are enclosed within <think>...</think> and <answer>...</answer> tags, "
        "respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. "
        f"User: {question} Assistant: {helper}"
    )
    return prompt

@torch.no_grad()
def generate_rollouts(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, questions: list[str], G: int, max_new_tokens: int, temperature: float):
    results =[]

    for question in questions:
        prompt = generate_prompt(question)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_len = inputs.input_ids.shape[1]

        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            num_return_sequences=G,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Build response mask (0 for prompt, 1 for response tokens)
        response_mask = torch.zeros_like(outputs, dtype=torch.float)
        response_mask[:, prompt_len:] = 1.0

        attention_mask = torch.ones_like(outputs, dtype=torch.float)

        # Decode response portions only
        response_texts = tokenizer.batch_decode(
            outputs[:, prompt_len:], skip_special_tokens=False
        )

        results.append({
            "question": question,
            "prompt_len": prompt_len,
            "input_ids": outputs,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "response_texts": response_texts,
        })

    return results


def estimate_vram(model: PreTrainedModel, config: GRPOConfig):
    """
    Estimate VRAM usage for GRPO training.
    Call after model initialization, before training.
    """
    def param_bytes(m):
        total = 0
        for p in m.parameters():
            total += p.numel() * p.element_size()
        return total

    def fmt(b):
        return f"{b / 1024**3:.2f} GB"

    model_bytes = param_bytes(model)
    hidden_size = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    vocab_size = model.config.vocab_size

    est_prompt_len = 128
    est_seq_len = est_prompt_len + config.max_new_tokens
    total_sequences = config.batch_size * config.G

    # ── Model ──
    model_mem = model_bytes                                     # π_θ
    ref_model_mem = model_bytes if config.use_kl else 0         # π_ref
    grad_mem = model_bytes                                      # gradients
    optimizer_mem = 2 * model_bytes                             # AdamW (m + v)

    # ── Cached log-probs (replaces old model copy) ──
    # Shape: [G, seq_len] per question, float32
    cached_logps_mem = config.batch_size * config.G * est_seq_len * 4  # float32

    # ── Rollout storage ──
    # input_ids (int64) + attention_mask (float32) + response_mask (float32)
    rollout_mem = total_sequences * est_seq_len * (8 + 4 + 4)  # bytes per element

    # ── Logits during forward pass ──
    # Shape: [batch_size * G, seq_len, vocab_size]
    logits_mem = total_sequences * est_seq_len * vocab_size * 2  # bfloat16

    # ── Activations for backward ──
    activation_mem = total_sequences * est_seq_len * hidden_size * 2 * num_layers * 2

    total = (
        model_mem + ref_model_mem +
        grad_mem + optimizer_mem +
        cached_logps_mem + rollout_mem +
        logits_mem + activation_mem
    )

    num_models = 2 if config.use_kl else 1

    print("=" * 55)
    print("VRAM Estimate")
    print("=" * 55)
    print(f"  Model (π_θ):             {fmt(model_mem)}")
    if config.use_kl:
        print(f"  Ref model (π_ref):       {fmt(ref_model_mem)}")
    print(f"  Gradients:               {fmt(grad_mem)}")
    print(f"  Optimizer (AdamW):        {fmt(optimizer_mem)}")
    print(f"  Cached log-probs:        {fmt(cached_logps_mem)}")
    print(f"  Rollout tensors:         {fmt(rollout_mem)}")
    print(f"    ({config.batch_size} questions × {config.G} rollouts × ~{est_seq_len} tokens)")
    print(f"  Logits:                  {fmt(logits_mem)}")
    print(f"  Activations (~est):      {fmt(activation_mem)}")
    print("-" * 55)
    print(f"  Total estimate:          {fmt(total)}")
    print("=" * 55)

    if torch.cuda.is_available():
        total_gpu = torch.cuda.get_device_properties(0).total_memory
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        print(f"\n  GPU total:               {fmt(total_gpu)}")
        print(f"  Currently allocated:     {fmt(allocated)}")
        print(f"  Currently reserved:      {fmt(reserved)}")
        print(f"  Free (approx):           {fmt(total_gpu - reserved)}")

        if total > total_gpu:
            print(f"\n  ⚠️  Estimated usage exceeds GPU memory by {fmt(total - total_gpu)}")
        else:
            print(f"\n  ✅ Should fit with ~{fmt(total_gpu - total)} headroom")

    return total