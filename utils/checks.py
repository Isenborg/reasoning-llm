import re
from .extracts import extract_answer

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

def check_single_thinking_block(text: str, tag: str = "think") -> bool:
    """
    Checks if the output contains exactly one thinking block.
    Returns True if exactly one opening and one closing tag exist.
    """
    open_tag = f"<{tag}>"
    close_tag = f"</{tag}>"
    
    return text.count(open_tag) == 1 and text.count(close_tag) == 1

def check_single_answer_block(text: str, tag: str = "answer") -> bool:
    """
    Checks if the output contains exactly one answer block.
    Returns True if exactly one opening and one closing tag exist.
    """
    open_tag = f"<{tag}>"
    close_tag = f"</{tag}>"
    
    return text.count(open_tag) == 1 and text.count(close_tag) == 1

def check_no_text_before_think(text: str, tag: str = "think") -> bool:
    """
    Checks that there is no text before the opening thinking tag.
    Allows leading whitespace or newlines.
    """
    open_tag = f"<{tag}>"
    tag_index = text.find(open_tag)
    
    # If the tag isn't found, the condition of "no text before it" is technically 
    # met, or we might want it to fail depending on our strictness. 
    if tag_index == -1:
        return False 
        
    # Check if a stripped version of the text before the tag is empty
    return text[:tag_index].strip() == ""

def is_correct_answer(text, ground_truth):
    answer = extract_answer(text)
    if answer is not None:
        return answer == ground_truth
    return False

