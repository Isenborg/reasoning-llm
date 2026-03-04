def extract_answer(text: str) -> str | None:
    """Extract the content from <answer>...</answer> tags."""
    import re
    match = re.search(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else None

def extract_thinking(text: str) -> str | None:
    import re
    match = re.search(r"<think>(.*?)</think>", text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else None

