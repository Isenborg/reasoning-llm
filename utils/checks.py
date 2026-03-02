import re

def has_complete_thinking_block(text: str) -> bool:
    """
    Returns True only if there is a <think> tag followed by a </think> tag.
    The block must not be empty.
    """
    pattern = r"<think>(?:.|\n)+?</think>"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return match is not None and len(match.group(0).replace("<think>", "").replace("</think>", "").strip()) > 0

def has_complete_answer_block(text: str) -> bool:
    """
    Returns True only if there is an <answer> tag followed by an </answer> tag.
    The block must not be empty.
    """
    pattern = r"<answer>(?:.|\n)+?</answer>"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return match is not None and len(match.group(0).replace("<answer>", "").replace("</answer>", "").strip()) > 0

def is_format_correct(text: str) -> bool:
    """
    Checks if both blocks exist and are properly closed.
    Useful for RLVR reward filtering.
    """
    return has_complete_thinking_block(text) and has_complete_answer_block(text)

def started_thinking_but_failed(text: str) -> bool:
    """
    Returns True if the model started a <think> block but never closed it
    or jumped straight to an <answer> tag.
    """
    has_start = bool(re.search(r"<think>", text, flags=re.IGNORECASE))
    has_end = bool(re.search(r"</think>", text, flags=re.IGNORECASE))
    return has_start and not has_end