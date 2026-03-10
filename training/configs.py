# configs/grpo_config.py
from dataclasses import dataclass

@dataclass
class GRPOConfig:
    G: int = 8                # Rollouts per question
    K: int = 1                # Gradient steps per rollout phase
    epsilon: float = 0.2      # Clip range
    lr: float = 1e-5
    max_steps: int = 100           # Number of training steps
    batch_size: int = 4       # Questions per batch
    max_new_tokens: int = 256
    temperature: float = 1.0

    # Kl penalty
    use_kl: bool = True
    beta: float = 0.01        # KL penalty weight (0 = no KL)

    # Eval
    eval_every: int = 20      # Steps between evals
    eval_samples: int = 50    # Number of samples to use for eval

    # Checkpoint management
    save_freq: int = -1      # Save every N steps. -1 for no checkpoints


@dataclass
class SFTConfig:
    lr: float = 5e-5
    epochs: int = 3
    batch_size: int = 2                  # Keep small to fit in VRAM
    grad_accum_steps: int = 4            # Effective batch = batch_size * grad_accum_steps
    max_length: int = 384                # Drop examples longer than this
    grad_clip: float = 1.0
    gradient_checkpointing: bool = True  # Trade compute for memory

    # Eval
    eval_every: int = 200       # Steps between evals (counts optimizer steps)
    eval_samples: int = 100     # Number of eval examples to use
