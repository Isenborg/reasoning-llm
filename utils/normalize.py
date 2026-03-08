import re

def gsm8k(x: str) -> str:
    x = x.lower().strip()
    x = x.replace(",", "")
    numbers = re.findall(r"-?\d+\.?\d*", x)

    if not numbers:
        return x

    # Take the last number found (usually the final answer in a thought chain)
    last_number = numbers[-1]
    
    return str(int(round(float(last_number))))