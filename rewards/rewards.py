from utils import extracts
from utils import checks
import re

def normalize(x: str) -> str:
    x = x.lower().strip()
    x = re.sub(r"[^\w\s]", "", x)
    x = re.sub(r"\s+", "", x)
    return x

def calculate_reward(
    text: str,
    ground_truth: str | None = None,
    format_weight: float = 0.25,
    correctness_weight: float = 0.75,
) -> float:

    reward = 0.0
    # Format reward
    is_single_think = checks.check_single_thinking_block(text)
    is_single_answer = checks.check_single_answer_block(text)
    no_preamble = checks.check_no_text_before_think(text)

    format_score = (
        0.08 * is_single_think +
        0.08 * is_single_answer +
        0.04 * no_preamble
    )

    reward += format_score * (format_weight / 0.25)
    think_block = extracts.extract_thinking(text)

    # Correctness reward
    if ground_truth is not None:
        gt = normalize(ground_truth)
        answer = extracts.extract_answer(text)
        if answer:
            if normalize(answer) == gt:
                reward += correctness_weight
            else:
                reward -= correctness_weight * 0.05
        else:
            reward -= correctness_weight * 0.1
        # Prevent answer leakage
        if think_block and gt in normalize(think_block):
            reward -= 0.1

    return max(0.0, min(reward, 1.0))