# configs/grpo_config.py
from dataclasses import dataclass

@dataclass
class GRPOConfig:
    G: int = 8                # Rollouts per question
    K: int = 1                # Gradient steps per rollout phase
    epsilon: float = 0.2      # Clip range
    lr: float = 1e-5
    max_steps: int = 1000           # Number of training steps
    batch_size: int = 4       # Questions per batch
    max_new_tokens: int = 256
    temperature: float = 1.0
    gradient_accumulation_steps: int = 8    # Effective batch size
    grad_clip: float = 2.0

    # Kl penalty
    use_kl: bool = True
    beta: float = 0.01        # KL penalty weight (0 = no KL)

    # Eval
    eval_every: int = 20      # Steps between evals
    eval_samples: int = 50    # Number of samples to use for eval

    # Warmup
    run_sft_warmup: bool = True

    # Checkpoint management
    save_freq: int = -1      # Save every N steps. -1 for no checkpoints

    # Potential 8 bit optimizer setting
    use_8bit_optim: bool = False
    use_wandb: bool = False


@dataclass
class SFTWarmupConfig:
    epochs: int = 3
    batch_size: int = 8
    lr: float = 5e-4
    grad_clip: float = 2.0
    log_every: int = 10


@dataclass
class SFTConfig:
    lr: float = 2e-5
    epochs: int = 3
    batch_size: int = 2                  # Keep small to fit in VRAM
    grad_accum_steps: int = 4            # Effective batch = batch_size * grad_accum_steps
    max_length: int = 1024                # Drop examples longer than this
    grad_clip: float = 1.0
    gradient_checkpointing: bool = True  # Trade compute for memory

    # LoRA
    use_lora: bool = True
    lora_r: int = 16            # Rank — higher = more capacity, more memory
    lora_alpha: int = 32        # Scaling factor (alpha/r = effective scale)
    lora_dropout: float = 0.05
    # Which linear layers to inject LoRA into (None = peft auto-detects all)
    lora_target_modules: list = None
    plot_training: bool = False

    # Eval
    eval_every: int = 100       # Steps between evals (counts optimizer steps)
    eval_samples: int = 128     # Number of eval examples to use

    use_8bit_optim: bool = False

