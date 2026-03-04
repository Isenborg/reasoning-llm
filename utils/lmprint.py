import re
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

console = Console()

def pretty_print(raw_output: str):
    # 1. Isolate the Assistant's response
    # We use rsplit to ensure we get the content after the VERY LAST "Assistant:"
    if "Assistant:" in raw_output:
        text = raw_output.rsplit("Assistant:", 1)[-1].strip()
    else:
        text = raw_output.strip()

    # Clean up end tokens
    text = text.replace("<|endoftext|>", "").strip()

    # 2. Split by tags
    tokens = re.split(r'(<think>|</think>|<answer>|</answer>)', text, flags=re.IGNORECASE)
    
    pending_text = []

    def flush_pending():
        """Helper to print any accumulated text as a standard response box."""
        nonlocal pending_text
        content = "".join(pending_text).strip()
        # Remove stray formatting characters often seen in base models
        content = content.strip('۰').strip()
        if content:
            render_panel(content, "📝 Response", "blue", "")
        pending_text = []

    i = 0
    while i < len(tokens):
        token = tokens[i]
        if not token:
            i += 1
            continue

        lower_token = token.lower()

        # --- ATTEMPT THINKING BLOCK ---
        if lower_token == "<think>":
            content, next_tag_idx = find_closing_tag(tokens, i + 1, "</think>", ["<answer>", "<think>"])
            if next_tag_idx != -1:
                flush_pending() # Print what came before
                render_panel(content, "🤔 Thinking", "yellow", "dim italic")
                i = next_tag_idx + 1
                continue
            else:
                # No closing tag? Treat the <think> tag itself as normal text and keep going
                pending_text.append(token)
                i += 1

        # --- ATTEMPT ANSWER BLOCK ---
        elif lower_token == "<answer>":
            content, next_tag_idx = find_closing_tag(tokens, i + 1, "</answer>", ["<think>", "<answer>"])
            if next_tag_idx != -1:
                flush_pending() # Print what came before
                render_panel(content, "💡 Answer", "green", "bold")
                i = next_tag_idx + 1
                continue
            else:
                pending_text.append(token)
                i += 1

        # --- STANDALONE TEXT OR BROKEN TAGS ---
        else:
            pending_text.append(token)
            i += 1

    # Final flush for any remaining text
    flush_pending()

def find_closing_tag(tokens, start_idx, target_tag, break_tags):
    accumulated = []
    for j in range(start_idx, len(tokens)):
        t = tokens[j]
        if t.lower() == target_tag:
            return "".join(accumulated).strip(), j
        if t.lower() in break_tags:
            return "".join(accumulated).strip(), -1
        accumulated.append(t)
    return "".join(accumulated).strip(), -1

def render_panel(content, title, color, style):
    if not content: return
    console.print(
        Panel(
            Markdown(content),
            title=f"[{color}]{title}[/{color}]",
            title_align="left",
            border_style=color,
            style=style,
            padding=(1, 2),
        )
    )

def print_question(question: str):
    """Print a question in a styled panel."""
    render_panel(question, "❓ Question", "cyan", "bold")