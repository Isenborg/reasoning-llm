from utils import extracts
from utils import checks

def compute_reward(text: str, ground_truth: str | None = None, **kwargs) -> float:
    """Compute reward for a generated completion (format + optional correctness).

    This is a thin wrapper around :func:`calculate_reward` that accepts the same
    inputs along with any additional keyword arguments used for weighting or
    future extensions.  It exists primarily so that callers can depend on a
    stable API without worrying about the internal implementation of the
    scoring logic.
    """
    # For now, simply forward to ``calculate_reward``.  Keyword arguments such
    # as ``format_weight`` and ``correctness_weight`` may be provided and will
    # be passed along directly.
    return calculate_reward(text, ground_truth, **kwargs)


def calculate_reward(text: str, ground_truth: str | None = None, 
                     format_weight: float = 0.3, correctness_weight: float = 0.7) -> float:
    reward = 0.0
    
    # Verify strict block constraints
    is_single_think = checks.check_single_thinking_block(text)
    is_single_answer = checks.check_single_answer_block(text)
    no_preamble = checks.check_no_text_before_think(text)
    
    # Format reward
    # Full reward requires all formatting rules to be perfect
    if checks.is_format_correct(text) and is_single_think and is_single_answer and no_preamble:
        reward += format_weight 
        
    # Partial reward for only an answer block (no_preamble doesn't apply here)
    elif checks.has_complete_answer_block(text) and is_single_answer:
        reward += format_weight * 0.5  
        
    # Partial reward for only a thinking block (requires no preamble)
    elif checks.has_complete_thinking_block(text) and is_single_think and no_preamble:
        reward += format_weight * 0.5  
    
    # Correctness reward
    if ground_truth is not None:
        answer = extracts.extract_answer(text)
        if answer and answer == ground_truth:
            reward += correctness_weight  # Correct answer
        elif answer:
            reward += correctness_weight * 0.1  # Small penalty for wrong answer
            
    return min(reward, 1.0)

