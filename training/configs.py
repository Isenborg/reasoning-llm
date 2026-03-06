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
    # TODO
    pass
