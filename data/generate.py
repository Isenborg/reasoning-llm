import random
import operator

# ============================================================
# Format generators — each returns (question, thinking, answer)
# ============================================================

OPS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
}

def format_simple_equation(rng: random.Random):
    """Solve x ⊕ y"""
    x = rng.randint(1, 500)
    y = rng.randint(1, 500)
    sign = rng.choice(["+", "-", "*"])
    result = OPS[sign](x, y)

    question = f"Solve: {x} {sign} {y}"
    thinking = f"{x} {sign} {y} = {result}"
    answer = str(result)
    return question, thinking, answer


def format_word_problem(rng: random.Random):
    """Word problem with a scenario"""
    templates = [
        {
            "setup": "{name} has {x} {item}s and {action} {y}. How many {item}s does {name} have now?",
            "actions": {
                "buys": operator.add,
                "finds": operator.add,
                "receives": operator.add,
                "loses": operator.sub,
                "gives away": operator.sub,
            },
            "items": ["apple", "coin", "book", "marble", "sticker", "card"],
            "names": ["Alice", "Bob", "Tom", "Sara", "Jack", "Maya"],
        },
        {
            "setup": "A store sells {item}s for ${x} each. {name} buys {y}. How much does {name} pay?",
            "op": operator.mul,
            "items": ["hat", "pen", "toy", "shirt", "mug"],
            "names": ["Alice", "Bob", "Tom", "Sara", "Jack", "Maya"],
        },
    ]

    template = rng.choice(templates)
    name = rng.choice(template["names"])
    item = rng.choice(template["items"])

    if "actions" in template:
        action, op = rng.choice(list(template["actions"].items()))
        x = rng.randint(10, 100)
        y = rng.randint(1, min(x, 50))  # avoid negatives for subtraction
        result = op(x, y)
        question = template["setup"].format(name=name, x=x, y=y, item=item, action=action)
        thinking = f"{name} starts with {x} {item}s. {action.capitalize()} {y}. {x} {'+' if op == operator.add else '-'} {y} = {result}."
        answer = str(result)
    else:
        x = rng.randint(2, 50)
        y = rng.randint(1, 10)
        result = template["op"](x, y)
        question = template["setup"].format(name=name, x=x, y=y, item=item)
        thinking = f"Each {item} costs ${x}. {name} buys {y}. {x} * {y} = {result}."
        answer = f"${result}"

    return question, thinking, answer


def format_multi_step(rng: random.Random):
    """Chained operations: (x ⊕ y) ⊗ z"""
    x = rng.randint(1, 50)
    y = rng.randint(1, 50)
    z = rng.randint(1, 20)
    sign1 = rng.choice(["+", "-"])
    sign2 = rng.choice(["*", "+", "-"])

    step1 = OPS[sign1](x, y)
    result = OPS[sign2](step1, z)

    question = f"Calculate: ({x} {sign1} {y}) {sign2} {z}"
    thinking = f"First: {x} {sign1} {y} = {step1}. Then: {step1} {sign2} {z} = {result}."
    answer = str(result)
    return question, thinking, answer


GENERATORS = [format_simple_equation, format_word_problem, format_multi_step]


# ============================================================
# Dataset that generates on the fly
# ============================================================
from torch.utils.data import Dataset
import torch

class RandomFormatSFTDataset(Dataset):
    def __init__(self, tokenizer, prompt_template, size=200, seed=42):
        self.samples = []
        rng = random.Random(seed)

        for _ in range(size):
            gen = rng.choice(GENERATORS)
            question, thinking, answer = gen(rng)

            prompt = f"{prompt_template(question)}\n"
            completion = f"<think>\n{thinking}\n</think>\n<answer>{answer}</answer>"
            full_text = prompt + completion

            full_ids = tokenizer.encode(full_text, add_special_tokens=False)
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

            labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

            self.samples.append({
                "input_ids": torch.tensor(full_ids),
                "labels": torch.tensor(labels),
                "attention_mask": torch.ones(len(full_ids)),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]