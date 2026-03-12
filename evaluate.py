#!/usr/bin/env python3
"""Interactive evaluation script with arrow-key navigable menus."""

import os
import logging
import warnings

os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)

import sys

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import IntPrompt, FloatPrompt, Prompt, Confirm
from rich import box
from simple_term_menu import TerminalMenu

from utils.models import DEFAULT_MODEL_DIR
from utils.menu import clear, model_selection_menu, pick_choice
from eval.eval_gsm8k import evaluate_gsm8k_vllm
from data.gsm8k import GSM8KDataset

console = Console()

TITLE = Panel(
    "[bold]GSM8K Model Evaluation[/]\n"
    "[dim]Interactive evaluation with majority-vote scoring[/]",
    border_style="magenta",
    padding=(1, 4),
)

PROMPT_STYLES = ["base", "reasoning"]


class HyperParams:
    FIELDS = [
        ("temperature",    "Temperature",     float,    "Sampling temperature"),
        ("maj_k",          "Majority-vote K", int,      "Samples for majority vote"),
        ("n_examples",     "Num examples",    int,      "Number of examples (-1 = all)"),
        ("max_new_tokens", "Max new tokens",  int,      "Max generated tokens"),
        ("prompt_style",   "Prompt style",    "choice", "Prompt template style"),
        ("dataset_split",  "Dataset split",   str,      "GSM8K split (test / train)"),
    ]

    CHOICES = {
        "prompt_style": PROMPT_STYLES,
    }

    def __init__(self):
        self.temperature: float = 1.0
        self.maj_k: int = 16
        self.n_examples: int = -1
        self.max_new_tokens: int = 1024
        self.prompt_style: str = "base"
        self.dataset_split: str = "test"

    def format_value(self, attr: str) -> str:
        val = getattr(self, attr)
        return "all" if attr == "n_examples" and val == -1 else str(val)


def hyperparameter_menu(model_str: str, hp: HyperParams) -> HyperParams | None:
    while True:
        clear(TITLE)
        console.print(f"  [bold cyan]Model:[/] {model_str}\n")
        console.rule("[bold magenta]Hyperparameters[/]")

        entries = []
        for attr, display, _, desc in hp.FIELDS:
            val_str = hp.format_value(attr)
            entries.append(f"  {display:<18s} = {val_str:<10s}  ({desc})")

        entries.append("")
        entries.append("▶  Run evaluation")
        entries.append("←  Back to model selection")

        run_idx = len(hp.FIELDS) + 1
        back_idx = run_idx + 1

        menu = TerminalMenu(
            entries,
            title="  ↑↓ navigate, Enter edit/select\n",
            skip_empty_entries=True,
        )
        idx = menu.show()

        if idx is None or idx == back_idx:
            return None
        if idx == run_idx:
            return hp

        attr, display, ftype, _ = hp.FIELDS[idx]
        current = getattr(hp, attr)

        try:
            if ftype == "choice":
                new_val = pick_choice(hp.CHOICES[attr], str(current), display)
            elif ftype is float:
                new_val = FloatPrompt.ask(f"  [bold]New {display}[/]", default=current)
            elif ftype is int:
                new_val = IntPrompt.ask(f"  [bold]New {display}[/]", default=current)
            else:
                new_val = Prompt.ask(f"  [bold]New {display}[/]", default=str(current))
            setattr(hp, attr, new_val)
        except KeyboardInterrupt:
            pass


def run_evaluation(model_str: str, hp: HyperParams):
    clear(TITLE)

    summary = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    summary.add_column(style="bold cyan")
    summary.add_column(style="white")
    summary.add_row("Model", model_str)
    for attr, display, _, _ in hp.FIELDS:
        summary.add_row(display, hp.format_value(attr))
    console.print(Panel(summary, title="Configuration", border_style="cyan"))

    console.print(f"[bold]Loading GSM8K ({hp.dataset_split})...[/]")
    dataset = GSM8KDataset(split=hp.dataset_split)

    evaluate_gsm8k_vllm(
        model_str,
        dataset,
        n_examples=hp.n_examples,
        maj_k=hp.maj_k,
        temperature=hp.temperature,
        max_new_tokens=hp.max_new_tokens,
        prompt_style=hp.prompt_style,
    )

    console.rule("[bold green]Evaluation Complete[/]")


def main():
    while True:
        model_str = model_selection_menu(TITLE)

        hp = HyperParams()
        result = hyperparameter_menu(model_str, hp)
        if result is None:
            continue

        run_evaluation(model_str, result)

        if not Confirm.ask("\n[bold]Run another evaluation?[/]", default=True):
            console.print("[dim]Goodbye![/]")
            break


if __name__ == "__main__":
    main()