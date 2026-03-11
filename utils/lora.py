"""
Minimal LoRA implementation — no external dependencies beyond PyTorch.

Injects trainable low-rank matrices (A, B) alongside frozen base weights:
    output = W·x  +  (B·A·x) * (alpha/r)

Only A and B are updated during training; everything else stays frozen.
"""

import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear and adds a trainable low-rank bypass."""

    def __init__(self, linear: nn.Linear, r: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        self.linear = linear
        self.r = r
        self.scale = alpha / r

        in_f = linear.in_features
        out_f = linear.out_features

        self.lora_A = nn.Parameter(torch.empty(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))  # zero-init → identity start
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Freeze base weight
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scale


def apply_lora(
    model: nn.Module,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: list[str] | None = None,
) -> nn.Module:
    """
    Freeze the entire model, then replace every linear layer whose name ends
    with one of the target_module suffixes with a LoRALinear.

    Default targets cover the attention projections used by Qwen / LLaMA style models.
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    replaced = 0
    for full_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(full_name.endswith(t) for t in target_modules):
            continue

        # Navigate to the parent and swap the child
        *parent_parts, child_name = full_name.split(".")
        parent = model
        for part in parent_parts:
            parent = getattr(parent, part)

        setattr(parent, child_name, LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
        replaced += 1

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA: replaced {replaced} layers | "
          f"{trainable:,} trainable / {total:,} total params ({100 * trainable / total:.2f}%)")

    return model


def save_lora(model: nn.Module, path) -> None:
    """Save only the LoRA adapter weights (A and B matrices)."""
    from pathlib import Path
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    adapter_state = {
        name: param
        for name, param in model.named_parameters()
        if "lora_A" in name or "lora_B" in name
    }
    torch.save(adapter_state, path / "lora_adapter.pt")
    print(f"LoRA adapter saved to {path / 'lora_adapter.pt'} "
          f"({sum(p.numel() for p in adapter_state.values()):,} params)")


def load_lora(model: nn.Module, path) -> nn.Module:
    """Load LoRA adapter weights back into a model that has already had apply_lora() called."""
    from pathlib import Path
    adapter_state = torch.load(Path(path) / "lora_adapter.pt", map_location="cpu")
    missing, unexpected = model.load_state_dict(adapter_state, strict=False)
    print(f"LoRA adapter loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    return model
