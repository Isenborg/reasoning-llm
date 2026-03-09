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

        pad_id = tokenizer.pad_token_id
        attention_mask = (outputs != pad_id).float()

        # Response mask: 1 only for actual response tokens (not prompt, not padding)
        response_mask = torch.zeros_like(outputs, dtype=torch.float)
        for i in range(outputs.shape[0]):
            seq_len = attention_mask[i].sum().long().item()
            response_mask[i, prompt_len:seq_len] = 1.0

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
    """Estimate peak VRAM usage for GRPO training."""

    def fmt(b):
        return f"{b / 1024**3:.2f} GB"

    # Model properties
    n_params = sum(p.numel() for p in model.parameters())
    dtype_size = next(model.parameters()).element_size()
    model_bytes = n_params * dtype_size

    mc = model.config
    hidden = mc.hidden_size
    n_layers = mc.num_hidden_layers
    vocab = mc.vocab_size
    n_kv_heads = getattr(mc, "num_key_value_heads", mc.num_attention_heads)
    head_dim = hidden // mc.num_attention_heads
    intermediate = getattr(mc, "intermediate_size", hidden * 4)

    seq_len = 128 + config.max_new_tokens
    G = config.G
    B = config.batch_size
    total_seq = B * G

    # ── Persistent (always in memory) ──
    weights = model_bytes
    ref = model_bytes if config.use_kl else 0
    grads = model_bytes
    optim = (n_params * 2) if config.use_8bit_optim else (model_bytes * 2)
    persistent = weights + ref + grads + optim

    # ── Phase peaks (only one active at a time) ──

    # Generation: KV cache for all sequences
    kv_cache = 2 * n_layers * total_seq * n_kv_heads * seq_len * head_dim * dtype_size

    # Cache log-probs: logits for G sequences at once
    cache_peak = G * seq_len * vocab * dtype_size

    # Training: 1 sequence at a time, grad checkpointing on
    train_logits = seq_len * vocab * dtype_size
    train_activations = (n_layers * seq_len * hidden * dtype_size  # checkpoint storage
                         + seq_len * intermediate * dtype_size)     # recompute buffer
    train_peak = train_logits + train_activations

    # Rollout tensors + cached logps (held during phases 2 and 3)
    rollout_storage = total_seq * seq_len * (8 + 8 + 4)  # ids, mask, response_mask
    cached_logps = total_seq * seq_len * 4                # float32

    # Peak = persistent + worst phase
    phase_peaks = {
        "Generation": kv_cache,
        "Cache log-probs": cache_peak + rollout_storage + cached_logps,
        "Training": train_peak + rollout_storage + cached_logps,
    }
    peak_phase = max(phase_peaks, key=phase_peaks.get)
    cuda_overhead = int(0.4 * 1024**3)
    peak_total = persistent + phase_peaks[peak_phase] + cuda_overhead

    # ── Print ──
    print("=" * 50)
    print("  VRAM ESTIMATE")
    print("=" * 50)
    print(f"  Model:       {n_params/1e9:.2f}B params, {dtype_size}B/param")
    print(f"  Sequences:   {B} × {G} = {total_seq}, ~{seq_len} tokens")
    print()
    print(f"  Weights:     {fmt(weights)}")
    if config.use_kl:
        print(f"  Ref model:   {fmt(ref)}")
    print(f"  Gradients:   {fmt(grads)}")
    label = "8-bit" if config.use_8bit_optim else "standard"
    print(f"  Optimizer:   {fmt(optim)} ({label})")
    print(f"  Persistent:  {fmt(persistent)}")
    print()
    for name, mem in phase_peaks.items():
        marker = " ← peak" if name == peak_phase else ""
        print(f"  {name:20s} +{fmt(mem)}{marker}")
    print()
    print(f"  Peak VRAM:   {fmt(peak_total)}")
    print("=" * 50)

    if torch.cuda.is_available():
        total_gpu = torch.cuda.get_device_properties(0).total_memory
        headroom = total_gpu - peak_total
        print(f"  GPU memory:  {fmt(total_gpu)}")
        if headroom > 0:
            print(f"  ✅ Fits with ~{fmt(headroom)} headroom")
        else:
            print(f"  ⚠️  Over by {fmt(-headroom)}")
        print()

    return peak_total