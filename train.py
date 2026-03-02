from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

print(f"Loading {MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype="bfloat16",
    device_map="auto",
)
print(f"Loaded. Device: {model.device}")

QUESTIONS = [
    # Easy — should get right
    ("What is 15 + 27?", "42"),

    # GSM8K-level
    ("Janet has 3 apples. She buys 2 more bags of apples, with 4 apples in each bag. "
     "How many apples does she have in total?", "11"),

    # Slightly harder
    ("A store sells notebooks for $4 each. If you buy 3 or more, you get a 25% discount. "
     "How much do 5 notebooks cost?", "15"),

    # Algebra
    ("Solve for x: 3x + 7 = 22", "5"),

    # Harder
    ("How many prime numbers are there between 1 and 30?", "10"),
]

SYSTEM = "You are a helpful math assistant. Solve the problem step by step. Put your final answer in \\boxed{}."

for question, expected in QUESTIONS:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
    )

    # Only decode the new tokens, not the prompt
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    print("=" * 60)
    print(f"Q: {question}")
    print(f"Expected: {expected}")
    print(f"Model output:\n{response}")
    print()