import re
from torch.utils.data import Dataset
from datasets import load_dataset


class GSM8KDataset(Dataset):
    """
    GSM8K dataset. Extracts final numeric answer from solution string.
    
    Raw answer format: "...#### 42"
    We extract: "42"
    """

    def __init__(self, split: str = "train"):
        raw = load_dataset("openai/gsm8k", "main", split=split)
        
        self.questions = []
        self.answers = []

        for example in raw:
            question = example["question"]
            answer = self._extract_answer(example["answer"])
            
            if answer is not None:
                self.questions.append(question)
                self.answers.append(answer)

        print(f"Loaded {len(self)} GSM8K examples ({split})")

    def _extract_answer(self, solution: str) -> str | None:
        """Extract final answer after #### marker."""
        match = re.search(r"####\s*(.+)", solution)
        if match:
            # Remove commas from numbers like "1,234"
            return match.group(1).strip().replace(",", "")
        return None

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return {
            "question": self.questions[idx],
            "answer": self.answers[idx],
        }