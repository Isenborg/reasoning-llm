import os
import sys

from rich.console import Console
from rich.prompt import Prompt
from simple_term_menu import TerminalMenu

from utils.models import DEFAULT_MODEL_DIR, list_models, download_model

console = Console()


def clear(title_panel):
    """Clear terminal and reprint the given title banner."""
    os.system("cls" if os.name == "nt" else "clear")
    console.print(title_panel)


def model_selection_menu(title_panel) -> str:
    """Arrow-key menu to pick a local model or download a new one."""
    while True:
        clear(title_panel)
        models = list_models()

        console.rule("[bold magenta]Select Model[/]")
        console.print(f"  [dim]{DEFAULT_MODEL_DIR}[/]\n")

        entries = []
        for m in models:
            entries.append(f"  {m}")

        entries.append("")
        entries.append("⬇  Download a new model")
        entries.append("✕  Quit")

        download_idx = len(models) + 1 if models else 1
        quit_idx = download_idx + 1

        menu = TerminalMenu(
            entries,
            title="  ↑↓ navigate, Enter select\n",
            skip_empty_entries=True,
            cursor_index=0 if models else download_idx,
        )
        idx = menu.show()

        if idx is None or idx == quit_idx:
            console.print("[dim]Goodbye![/]")
            sys.exit(0)

        if idx == download_idx:
            hf_name = Prompt.ask(
                "\n[bold cyan]HuggingFace model name[/] "
                "[dim](e.g. Qwen/Qwen3-1.7B-Base)[/]"
            )
            if hf_name.strip():
                try:
                    with console.status(
                        f"[bold cyan]Downloading {hf_name.strip()}...[/]",
                        spinner="dots",
                    ):
                        download_model(hf_name.strip())
                    console.print(f"[bold green]✓[/] Downloaded {hf_name.strip()}")
                except Exception as e:
                    console.print(f"[bold red]✗ Error:[/] {e}")
                Prompt.ask("[dim]Press Enter to continue[/]")
            continue

        if idx < len(models):
            return models[idx]


def pick_choice(options: list[str], current: str, label: str = "Pick a value") -> str:
    """Arrow-key sub-menu for picking from a fixed set of options."""
    cursor = options.index(current) if current in options else 0
    menu = TerminalMenu(
        [f"  {opt}" for opt in options],
        title=f"\n  {label} (current: {current})\n",
        cursor_index=cursor,
    )
    idx = menu.show()
    return current if idx is None else options[idx]