from utils import extracts
from utils import checks
import re

def normalize_gsm8k(x: str) -> str:
    x = x.lower().strip()
    x = x.replace(",", "")
    numbers = re.findall(r"-?\d+\.?\d*", x)

    if not numbers:
        return x

    # Take the last number found (usually the final answer in a thought chain)
    last_number = numbers[-1]
    
    return str(int(round(float(last_number))))


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
        0.1 * is_single_think +
        0.1 * is_single_answer +
        0.05 * no_preamble
    )

    reward += format_score

    # Correctness reward
    if ground_truth is not None:
        gt = normalize_gsm8k(ground_truth)
        answer = extracts.extract_answer(text)
        if answer and normalize_gsm8k(answer) == gt:
            reward += correctness_weight

    return reward