from data.gsm8k import GSM8KSFTDataset
from torch.utils.data import random_split

from utils.models import load_tokenizer, load_model
from grpo.utils import generate_prompt
from training.configs import SFTConfig
from training.sft import SFTTrainer

model_str = "Qwen/Qwen3-1.7B-Base"

model = load_model(model_str)
tokenizer = load_tokenizer(model_str)

dataset = GSM8KSFTDataset(tokenizer, generate_prompt)
print(f"Longest sequence: {dataset.longest_sequence()} tokens")
print(f"Total number of tokens: {dataset.total_tokens()} tokens")

# Hold out 15% for eval
eval_size = int(0.15 * len(dataset))
train_size = len(dataset) - eval_size
train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

print(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

config = SFTConfig(
    batch_size=4,
    grad_accum_steps=4,
    epochs=1,
)

trainer = SFTTrainer(model, tokenizer, config)
trainer.train(train_dataset=train_dataset, eval_dataset=eval_dataset, run_name="sft_training")