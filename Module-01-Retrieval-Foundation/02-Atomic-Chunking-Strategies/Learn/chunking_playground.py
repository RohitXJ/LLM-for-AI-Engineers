import os
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

def visualize_chunks(chunks, title):
    """Prints chunks with alternating colors for easy visualization."""
    console.rule(f"[bold blue]{title}[/bold blue]")
    for i, chunk in enumerate(chunks):
        color = "cyan" if i % 2 == 0 else "green"
        console.print(Panel(
            Text(chunk, style=color),
            title=f"Chunk {i+1} ({len(chunk)} chars)",
            border_style="white"
        ))

# Sample text: A bit of lore about AI
SAMPLE_TEXT = """Artificial Intelligence (AI) has undergone a dramatic transformation. 
Early AI systems were "Rule-Based," meaning they followed strict If-Then logic. 
Imagine a chess engine that knows every possible move but doesn't "learn" from its mistakes.

Then came "Machine Learning" (ML). Instead of being told the rules, the system was given data. 
It looked for patterns. It failed, it adjusted, it improved. 

Today, we are in the era of "Large Language Models" (LLMs). 
These systems aren't just looking for patterns; they are predicting the next token in a sequence 
based on trillions of parameters. This is where "Chunking" becomes critical. 
If we feed an LLM too much information, it loses the needle in the haystack. 
If we feed it too little, it lacks the context to understand the question."""

def main():
    chunk_size = 200
    chunk_overlap = 50

    # 1. NAIVE CHARACTER SPLITTING
    # ----------------------------
    # Logic: Cut exactly at the character limit, regardless of words.
    char_splitter = CharacterTextSplitter(
        separator="",
        chunk_size=chunk_size,
        chunk_overlap=0,
    )
    char_chunks = char_splitter.split_text(SAMPLE_TEXT)
    visualize_chunks(char_chunks, "Naive Character Splitting (No Overlap)")

    # 2. RECURSIVE CHARACTER SPLITTING
    # --------------------------------
    # Logic: Tries to split on paragraphs (\n\n), then newlines (\n), then spaces.
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    rec_chunks = recursive_splitter.split_text(SAMPLE_TEXT)
    visualize_chunks(rec_chunks, "Recursive Splitting (With 50 char Overlap)")

    # PRO TIP: Check the overlap
    console.print("\n[bold yellow]Senior AI Engineer Insight:[/bold yellow]")
    console.print("Notice how Chunk 2 starts with the end of Chunk 1 in the Recursive example.")
    console.print("This overlap ensures that if a question relates to the transition between chunks,")
    console.print("the LLM has enough context from both sides to answer accurately.")

if __name__ == "__main__":
    main()
