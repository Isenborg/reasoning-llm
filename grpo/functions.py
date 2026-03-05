import torch
import torch.nn.functional as F
from transformers import PreTrainedModel


def get_per_token_logps(model: PreTrainedModel, input_ids: torch.LongTensor, attention_mask: torch.LongTensor) -> torch.FloatTensor:
    """
    Returns the log probabilities for generating each output token, based on the previous tokens.
    Shape: [batch, seq_len]
    """
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    # logits[:, t, :] predicts token t+1, so we only need up to t-1
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)

    token_logps = log_probs.gather(
        dim=-1,
        index=input_ids[:, 1:].unsqueeze(-1),
    ).squeeze(-1)  # [batch, seq_len - 1]

    # Pad so output aligns with input_ids shape
    token_logps = F.pad(token_logps, pad=(1, 0), value=0.0)

    return token_logps


def compute_ratio(model_new, model_old, input_ids, attention_mask, response_mask):
    """
    Probability ratio π_θ / π_θ_old, per token.
    """
    new_logps = get_per_token_logps(model_new, input_ids, attention_mask)

    with torch.no_grad():
        old_logps = get_per_token_logps(model_old, input_ids, attention_mask)

    log_ratio = (new_logps - old_logps) * response_mask
    return torch.exp(log_ratio)


def compute_ratio_from_logps(model_new, old_logps, input_ids, attention_mask, response_mask):
    """
    Ratio using cached old log-probs instead of a full model copy.
    """
    new_logps = get_per_token_logps(model_new, input_ids, attention_mask)
    log_ratio = (new_logps - old_logps) * response_mask
    return torch.exp(log_ratio)


def compute_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """
    Per-question advantage normalization.
    rewards: [G]
    """
    return (rewards - rewards.mean()) / (rewards.std() + 1e-8)


def clipped_surrogate_objective(ratios, advantages, epsilon):
    """
    min(r * A, clip(r, 1-ε, 1+ε) * A)
    """
    unclipped = ratios * advantages
    clipped = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * advantages
    return torch.min(unclipped, clipped)


def compute_kl_penalty(model_new, model_ref, input_ids, attention_mask, response_mask):
    """
    Per-token KL divergence estimate, only on response tokens.
    """
    new_logps = get_per_token_logps(model_new, input_ids, attention_mask)

    with torch.no_grad():
        ref_logps = get_per_token_logps(model_ref, input_ids, attention_mask)

    log_ratio = ref_logps - new_logps
    kl = torch.exp(log_ratio) - log_ratio - 1

    return kl * response_mask


def grpo_loss(ratios, advantages, response_mask, epsilon=0.2):
    """
    L = -1/G Σ_i 1/|o_i| Σ_t min(r * A, clip(r) * A)
    """
    adv = advantages.unsqueeze(1)                                       # [G] → [G, 1]
    surrogate = clipped_surrogate_objective(ratios, adv, epsilon)       # [G, seq_len]

    token_counts = response_mask.sum(dim=1).clamp(min=1)                # [G]
    per_output = (surrogate * response_mask).sum(dim=1) / token_counts  # [G]

    return -per_output.mean()


def grpo_loss_with_kl(ratios, advantages, kl, response_mask, epsilon=0.2, beta=0.01):
    """
    L = -1/G Σ_i 1/|o_i| Σ_t (min(r * A, clip(r) * A) - β * KL)
    """
    adv = advantages.unsqueeze(1)
    surrogate = clipped_surrogate_objective(ratios, adv, epsilon)

    penalized = surrogate - beta * kl

    token_counts = response_mask.sum(dim=1).clamp(min=1)
    per_output = (penalized * response_mask).sum(dim=1) / token_counts

    return -per_output.mean()