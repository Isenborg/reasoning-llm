from utils import extracts, checks, normalize
import re


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
        gt = normalize.gsm8k(ground_truth)
        answer = extracts.extract_answer(text)
        if answer and normalize.gsm8k(answer) == gt:
            reward += correctness_weight

    return reward