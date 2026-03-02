# utils.py
from transformers import TextStreamer
from rich.console import Console

class PrettyStreamer(TextStreamer):
    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.console = Console(force_terminal=True)
        self.in_think = False

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Called when a new chunk of text is ready to be printed."""
        # Check for tags in the text chunk
        if "<think>" in text:
            self.in_think = True
            # Print the header for thinking
            self.console.print("\n[bold cyan]🤔 THOUGHT PROCESS:[/bold cyan]")
            text = text.replace("<think>", "")

        if "</think>" in text:
            self.in_think = False
            # Print the header for the answer
            text = text.replace("</think>", "")
            self.console.print("\n\n[bold green]✅ FINAL ANSWER:[/bold green]")

        # Style the text based on current state
        if self.in_think:
            self.console.print(f"[italic cyan]{text}[/italic cyan]", end="")
        else:
            self.console.print(text, end="")

        if stream_end:
            self.console.print("\n" + "—"*30 + "\n")