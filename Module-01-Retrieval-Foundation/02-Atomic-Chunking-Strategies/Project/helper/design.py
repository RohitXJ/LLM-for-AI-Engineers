from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.align import Align

# Create a globally accessible colorful Console instance
console = Console()

def print_banner():
    """Prints a highly colorful, professional Gemini-CLI style banner."""
    title = Text("✧  CHUNKER-PRO CLI  ✧", style="bold bright_cyan on black", justify="center")
    
    subtitle = Text("Advanced RAG Pre-processor & Atomic Chunking Engine", style="italic medium_purple4", justify="center")
    
    agent_info = (
        "\n[bold bright_green]Agent Role:[/bold bright_green] [white]Semantic & Token-aware Document Processor[/white]\n"
        "[bold bright_green]Objective:[/bold bright_green] [white]Transform unstructured text into high-fidelity knowledge chunks.[/white]\n"
        "[bold bright_green]Architecture:[/bold bright_green] [white]Supports Recursive, Token-based, and Semantic splitting strategies.[/white]"
    )
    
    panel = Panel(
        Align.center(title + "\n\n" + subtitle + "\n" + Text.from_markup(agent_info)),
        border_style="bold bright_blue",
        padding=(1, 2),
    )
    console.print(panel)
    console.print()

def print_status_message(message, style="bold bright_cyan", icon="ℹ"):
    """Prints a colorful standard status or info message."""
    console.print(f"[{style}]{icon}[/{style}] {message}")

def print_success_message(message):
    """Prints a colorful success message."""
    console.print(f"[bold bright_green]✔[/bold bright_green] {message}\n")

def print_error_message(message):
    """Prints a colorful error message."""
    console.print(f"\n[bold bright_red]✖ Error:[/bold bright_red] {message}")

def get_spinner(message, spinner_style="dots"):
    """Returns a rich status spinner context manager."""
    return console.status(f"[bold bright_green]{message}", spinner=spinner_style)

def print_summary_table(file_path, strategy, docs_len, chunks_len, total_chars, total_tokens, avg_chars, avg_tokens):
    """Prints a colorful execution summary table."""
    table = Table(title="📊 Final Processing Summary", show_header=True, header_style="bold bright_magenta")
    table.add_column("Metric", style="bright_cyan", justify="left")
    table.add_column("Value", style="bold bright_white", justify="right")
    
    table.add_row("Input File", file_path)
    table.add_row("Chunking Strategy", strategy.capitalize())
    table.add_row("Total Initial Units", str(docs_len))
    table.add_row("Total Chunks Generated", str(chunks_len))
    table.add_row("Total Characters", f"{total_chars:,}")
    table.add_row("Total Tokens (o200k_base)", f"{total_tokens:,}")
    table.add_row("Average Chars / Chunk", f"{avg_chars:.1f}")
    table.add_row("Average Tokens / Chunk", f"{avg_tokens:.1f}")
    
    console.print(Align.center(table))
    console.print("\n[italic bright_black]System ready for next task. Terminating process.[/italic bright_black]\n")

tqdm_bar_format = "{l_bar}{bar:40}{r_bar}"
