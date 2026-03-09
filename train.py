import torch
from training.configs import GRPOConfig, SFTWarmupConfig
from training.grpo import GRPOTrainer
from training.sft_warmup import sft_warmup
from grpo import utils
from utils import models
from data.gsm8k import GSM8KDataset
from rewards import rewards

# ── Model ──
model_name = "Qwen/Qwen3-1.7B-Base"
run_name = "grpo-qwen3-gsm8k-run2"

model = models.load_model(model_name)
tokenizer = models.load_tokenizer(model_name)
print(f"Model dtype: {model.dtype}")

# ── Config ──
config = GRPOConfig(
    G=16,
    batch_size=2,
    K=2,
    lr=5e-6,
    epsilon=0.2,
    temperature=1.0,
    max_new_tokens=512,
    max_steps=5000,
    use_8bit_optim=True,
    use_wandb=False,
    use_kl=False,
    run_sft_warmup=True,
    eval_every=100,
    eval_samples=64,
    save_freq=500
)

# ── VRAM check ──
utils.estimate_vram(model, config)

# ── Datasets ──
train_dataset = GSM8KDataset(split="train")  # ~7.5k examples
test_dataset = GSM8KDataset(split="test")     # ~1.3k examples
print(f"Train: {len(train_dataset)} | Eval: {len(test_dataset)}")

# ── SFT warmup ──
if config.run_sft_warmup:
    from data.generate import RandomFormatSFTDataset

    warmup_config = SFTWarmupConfig()
    sft_dataset = RandomFormatSFTDataset(tokenizer, utils.generate_prompt, size=100)
    sft_warmup(
        model=model,
        tokenizer=tokenizer,
        dataset=sft_dataset,
        config=warmup_config
    )
    del sft_dataset

# ── GRPO training ──
trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    reward_fn=rewards.calculate_reward,
    config=config,
)

trainer.train(
    dataset=train_dataset,
    eval_dataset=test_dataset,
    run_name=run_name,
    run_id=run_name
)