import os
import datetime
import time
from pathlib import Path
from PyPDF2 import PdfReader
import docx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from contextlib import contextmanager

console = Console()

# TQDM styling
tqdm_bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"

def print_banner():
    """Prints a professional CLI header."""
    console.print(Panel.fit(
        "[bold cyan]CHUNKER-PRO CLI[/bold cyan]\n"
        "[gray]The Advanced RAG Pre-processor[/gray]",
        border_style="green",
        padding=(1, 2)
    ))

def print_status_message(message: str):
    console.print(f"[bold blue]INFO:[/bold blue] {message}")

def print_success_message(message: str):
    console.print(f"\n[bold green]✔ SUCCESS:[/bold green] {message}")

def print_error_message(message: str):
    console.print(f"\n[bold red]✘ ERROR:[/bold red] {message}")

@contextmanager
def get_spinner(text: str, spinner_style: str = "dots"):
    with console.status(text, spinner=spinner_style) as status:
        yield status

def extract_text(file_path: str) -> list[dict]:
    """
    Extracts text and returns a list of 'Document' dictionaries.
    Each dict: {"content": str, "metadata": dict}
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # File-level metadata
    file_stat = path.stat()
    mod_date = datetime.datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
    
    base_metadata = {
        "source": path.name,
        "extension": ext,
        "last_modified": mod_date
    }

    documents = []

    try:
        if ext in [".txt", ".md"]:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                documents.append({
                    "content": content,
                    "metadata": {**base_metadata, "page": 1}
                })

        elif ext == ".pdf":
            reader = PdfReader(file_path)
            for i, page in enumerate(reader.pages):
                content = page.extract_text()
                if content and content.strip():
                    documents.append({
                        "content": content,
                        "metadata": {**base_metadata, "page": i + 1}
                    })

        elif ext == ".docx":
            doc = docx.Document(file_path)
            pages = []
            current_page_text = []
            
            for para in doc.paragraphs:
                if 'lastRenderedPageBreak' in para._element.xml or 'w:br' in para._element.xml and 'type="page"' in para._element.xml:
                    if current_page_text:
                        pages.append("\n".join(current_page_text))
                    current_page_text = [para.text]
                else:
                    current_page_text.append(para.text)
            
            if current_page_text:
                pages.append("\n".join(current_page_text))

            for i, page_content in enumerate(pages):
                if page_content.strip():
                    documents.append({
                        "content": page_content,
                        "metadata": {**base_metadata, "page": i + 1}
                    })
        else:
            # Silently ignore unsupported files when batching, or raise for single file
            raise ValueError(f"Unsupported file extension: {ext}")

        return documents

    except Exception as e:
        raise Exception(f"Error processing {file_path}: {str(e)}")

def print_summary_table(file_path, strategy, docs_len, chunks_len, total_chars, total_tokens, avg_chars, avg_tokens):
    """Prints a detailed summary table of the chunking results."""
    table = Table(title="[bold]Chunking Results Summary[/bold]", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bright_white")

    table.add_row("Source", str(file_path))
    table.add_row("Strategy", strategy.upper())
    table.add_row("Input Documents", str(docs_len))
    table.add_row("Total Chunks", str(chunks_len))
    table.add_row("Total Characters", f"{total_chars:,}")
    table.add_row("Total Tokens", f"{total_tokens:,}")
    table.add_row("Avg Chars/Chunk", f"{avg_chars:.2f}")
    table.add_row("Avg Tokens/Chunk", f"{avg_tokens:.2f}")

    console.print("\n")
    console.print(table)
