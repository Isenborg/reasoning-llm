from training.configs import GRPOConfig
from training.grpo import GRPOTrainer
from grpo import utils
from utils import models

model_name = "grpo_model-best"

model = models.load_model(model_name)
tokenizer = models.load_tokenizer(model_name)

config = GRPOConfig(G=8, 
                    max_steps=200, 
                    batch_size=4, 
                    use_kl=False, 
                    eval_samples=16, 
                    eval_every=50, 
                    K=1, 
                    lr=5e-6, 
                    max_new_tokens=512, 
                    use_8bit_optim=True)

utils.estimate_vram(model, config)


from data.gsm8k import GSM8KDataset
from rewards import rewards

# Training
train_dataset = GSM8KDataset(split="train")   # ~7.5k examples
# Evaluation
test_dataset = GSM8KDataset(split="test")      # ~1.3k examples

trainer = GRPOTrainer(model, tokenizer, rewards.calculate_reward, config)
trainer.train(train_dataset, test_dataset)  