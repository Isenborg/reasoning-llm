#!/usr/bin/env python3
"""Interactive chat with a local model."""

import os
import random
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import gc

from rich.console import Console
from rich.panel import Panel
from rich.prompt import FloatPrompt, IntPrompt
from rich.table import Table
from rich import box
from simple_term_menu import TerminalMenu

from utils.models import load_model, load_tokenizer, DEFAULT_MODEL_DIR
from utils.menu import clear, model_selection_menu, pick_choice
from data.gsm8k import GSM8KDataset

console = Console()

TITLE = Panel(
    "[bold]Model Chat[/]\n"
    "[dim]Interactive chat with a local model[/]",
    border_style="blue",
    padding=(1, 4),
)

PROMPT_STYLES = ["base", "reasoning"]


# ── Prompts ──────────────────────────────────────────────────────────────────


def generate_prompt_reasoning(
    question,
    think_start="<think>",
    think_stop="</think>",
    answer_start="<answer>",
    answer_stop="</answer>",
):
    return (
        "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
        "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
        f"The reasoning process and answer are enclosed within {think_start}...{think_stop} and {answer_start}...{answer_stop} tags, "
        f"respectively, i.e., {think_start} reasoning process here {think_stop} {answer_start} answer here {answer_stop}. "
        f"User: {question}. Assistant: "
    )


def generate_prompt_base(
    question,
    answer_start="<answer>",
    answer_stop="</answer>",
):
    return (
        "A conversation between User and Assistant. The user asks a math question, and the Assistant solves it. "
        f"The answer is enclosed within {answer_start}...{answer_stop} tags, "
        f"i.e., {answer_start} answer here {answer_stop}. "
        f"User: {question}. Assistant: "
    )


def build_prompt(question: str, style: str) -> str:
    if style == "reasoning":
        return generate_prompt_reasoning(question)
    return generate_prompt_base(question)


# ── Settings ─────────────────────────────────────────────────────────────────


class ChatSettings:
    FIELDS = [
        ("temperature",    "Temperature",     float,    "Sampling temperature"),
        ("max_new_tokens", "Max new tokens",  int,      "Max generated tokens"),
        ("prompt_style",   "Prompt style",    "choice", "Prompt template style"),
    ]

    CHOICES = {
        "prompt_style": PROMPT_STYLES,
    }

    def __init__(self):
        self.temperature: float = 0.7
        self.max_new_tokens: int = 1024
        self.prompt_style: str = "base"

    def format_value(self, attr: str) -> str:
        return str(getattr(self, attr))


def settings_menu(model_str: str, settings: ChatSettings) -> ChatSettings | None:
    """Arrow-key menu to tweak chat settings."""
    while True:
        clear(TITLE)
        console.print(f"  [bold cyan]Model:[/] {model_str}\n")
        console.rule("[bold blue]Settings[/]")

        entries = []
        for attr, display, _, desc in settings.FIELDS:
            val_str = settings.format_value(attr)
            entries.append(f"  {display:<18s} = {val_str:<10s}  ({desc})")

        entries.append("")
        entries.append("▶  Start chat")
        entries.append("←  Back to model selection")

        start_idx = len(settings.FIELDS) + 1
        back_idx = start_idx + 1

        menu = TerminalMenu(
            entries,
            title="  ↑↓ navigate, Enter edit/select\n",
            skip_empty_entries=True,
        )
        idx = menu.show()

        if idx is None or idx == back_idx:
            return None
        if idx == start_idx:
            return settings

        attr, display, ftype, _ = settings.FIELDS[idx]
        current = getattr(settings, attr)

        try:
            if ftype == "choice":
                new_val = pick_choice(settings.CHOICES[attr], str(current), display)
            elif ftype is float:
                new_val = FloatPrompt.ask(f"  [bold]New {display}[/]", default=current)
            elif ftype is int:
                new_val = IntPrompt.ask(f"  [bold]New {display}[/]", default=current)
            else:
                from rich.prompt import Prompt
                new_val = Prompt.ask(f"  [bold]New {display}[/]", default=str(current))
            setattr(settings, attr, new_val)
        except KeyboardInterrupt:
            pass


# ── Streaming Generation ─────────────────────────────────────────────────────


def stream_generate(model, tokenizer, prompt: str, settings: ChatSettings):
    """Generate tokens one at a time with styled streaming output.

    - <think>...</think>  → dim grey text
    - <answer>...</answer> → bold text
    - everything else     → normal text
    """
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    temp = settings.temperature

    # State machine: "normal", "thinking", "answering"
    state = "normal"
    # Buffer for detecting tags
    tag_buffer = ""
    full_response = ""

    TAGS = {
        "<think>": "thinking",
        "</think>": "normal",
        "<answer>": "answering",
        "</answer>": "normal",
    }
    MAX_TAG_LEN = max(len(t) for t in TAGS)

    generated_ids = input_ids.clone()

    for _ in range(settings.max_new_tokens):
        with torch.no_grad():
            outputs = model(generated_ids)
            logits = outputs.logits[:, -1, :]

            if temp > 0:
                probs = torch.softmax(logits / temp, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = logits.argmax(dim=-1, keepdim=True)

        generated_ids = torch.cat([generated_ids, next_id], dim=-1)
        token_str = tokenizer.decode(next_id[0], skip_special_tokens=False)

        if tokenizer.eos_token and token_str.strip() == tokenizer.eos_token.strip():
            break

        full_response += token_str
        tag_buffer += token_str

        # Try to flush the buffer
        while tag_buffer:
            # Check if buffer starts with a complete tag
            matched = False
            

            for tag, new_state in TAGS.items():
                if tag_buffer.startswith(tag):
                    state = new_state
                    tag_buffer = tag_buffer[len(tag):]
                    # Style the tag markers
                    if tag == "<think>":
                        console.print("\n[dim italic]💭 Thinking...[/]")
                    elif tag == "<answer>":
                        console.print("\n[bold green]Answer:[/] ", end="")
                    elif tag == "</answer>":
                        console.print()  # newline after answer
                    matched = True
                    break

            if matched:
                continue

            # Check if buffer could be the start of any tag
            could_be_tag = any(
                tag.startswith(tag_buffer) for tag in TAGS
            )

            if could_be_tag:
                # Wait for more tokens
                break

            # Not a tag — flush first character
            ch = tag_buffer[0]
            tag_buffer = tag_buffer[1:]
            _print_char(ch, state)

    # Flush remaining buffer
    for ch in tag_buffer:
        _print_char(ch, state)

    console.print()  # final newline
    return full_response


def _print_char(ch: str, state: str):
    """Print a single character with style based on current state."""
    if state == "thinking":
        console.print(f"[dim italic]{ch}[/]", end="", highlight=False)
    elif state == "answering":
        console.print(f"[bold green]{ch}[/]", end="", highlight=False)
    else:
        console.print(ch, end="", highlight=False)


# ── Chat Loop ────────────────────────────────────────────────────────────────


def print_header(settings: ChatSettings):
    """Print the chat header with settings and commands."""
    console.print(
        f"  [dim]temp={settings.temperature}  "
        f"style={settings.prompt_style}  "
        f"max_tokens={settings.max_new_tokens}[/]"
    )
    console.rule("[bold blue]Chat[/]")

    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column(style="bold yellow", width=12)
    table.add_column(style="dim")
    table.add_row("/random /r", "Random GSM8K question")
    table.add_row("/clear  /c", "Clear conversation")
    table.add_row("/set    /s", "Back to settings")
    table.add_row("/quit   /q", "Quit")
    console.print(table)


def chat_loop(model, tokenizer, settings: ChatSettings, dataset) -> str:
    """Main chat loop. Returns 'settings' or 'quit'."""
    clear(TITLE)
    print_header(settings)

    while True:
        try:
            user_input = console.input("\n[bold green]You:[/] ").strip()
        except (KeyboardInterrupt, EOFError):
            return "quit"

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd in ("/quit", "/q"):
            return "quit"

        if cmd in ("/set", "/s"):
            return "settings"

        if cmd in ("/clear", "/c"):
            clear(TITLE)
            print_header(settings)
            console.print("[dim]Conversation cleared.[/]")
            continue

        expected = None
        if cmd in ("/random", "/r"):
            item = dataset[random.randint(0, len(dataset) - 1)]
            question = item["question"]
            answer = item["answer"]
            expected = answer.split("####")[-1].strip() if "####" in answer else answer

            console.print(f"\n[bold yellow]GSM8K Question:[/]")
            console.print(Panel(question, border_style="yellow"))

            user_input = question

        prompt = build_prompt(user_input, settings.prompt_style)

        console.print(f"\n[bold cyan]Assistant:[/] ", end="")
        response = stream_generate(model, tokenizer, prompt, settings)

        if expected:
            console.print(f"\n[bold yellow]Expected answer:[/] {expected}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    dataset = GSM8KDataset(split="test")

    while True:
        model_str = model_selection_menu(TITLE)
        settings = ChatSettings()

        clear(TITLE)
        with console.status("[bold cyan]Loading model...[/]", spinner="dots"):
            model = load_model(model_str)
            tokenizer = load_tokenizer(model_str)

        try:
            while True:
                result = settings_menu(model_str, settings)
                if result is None:
                    break  # → model selection

                action = chat_loop(model, tokenizer, settings, dataset)
                if action == "quit":
                    console.print("\n[dim]Goodbye![/]")
                    return
                # action == "settings" → loop back
        finally:
            import torch
            del model, tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()


if __name__ == "__main__":
    main()