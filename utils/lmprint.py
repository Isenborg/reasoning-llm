import re
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule

console = Console()

def response(raw_output: str):
    """
    Parses an LLM output containing <think> and <answer> tags
    and renders it beautifully using Rich.
    """

    think_match = re.search(r'<think>(.*?)</think>', raw_output, flags=re.DOTALL | re.IGNORECASE)
    answer_match = re.search(r'<answer>(.*?)</answer>', raw_output, flags=re.DOTALL | re.IGNORECASE)

    # --- 1. Thinking Section ---
    if think_match:
        think_text = think_match.group(1).strip()

        console.print()
        console.print(
            Panel(
                Markdown(think_text),
                title="[dim italic]🤔 Thinking...[/dim italic]",
                title_align="left",
                border_style="dim",
                style="dim italic",
                padding=(1, 2),
            )
        )

    # --- 2. Answer Section ---
    if answer_match:
        answer_text = answer_match.group(1).strip()

        console.print()
        console.print(
            Panel(
                Markdown(answer_text),
                title="[bold green]💡 Answer[/bold green]",
                title_align="left",
                border_style="green",
                padding=(1, 2),
            )
        )
        console.print()

    # --- 3. Fallbacks ---
    elif think_match and not answer_match:
        # Grab whatever text exists outside the <think> tags
        leftover = re.sub(r'<think>.*?</think>', '', raw_output, flags=re.DOTALL | re.IGNORECASE).strip()
        if leftover:
            console.print()
            console.print(
                Panel(
                    Markdown(leftover),
                    title="[bold green]💡 Answer[/bold green]",
                    title_align="left",
                    border_style="green",
                    padding=(1, 2),
                )
            )
            console.print()

    elif not think_match and not answer_match:
        # No tags found — just render the raw output as Markdown
        console.print()
        console.print(
            Panel(
                Markdown(raw_output.strip()),
                title="[bold blue]📝 Response[/bold blue]",
                title_align="left",
                border_style="blue",
                padding=(1, 2),
            )
        )
        console.print()