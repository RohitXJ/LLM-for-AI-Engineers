import os
from dotenv import load_dotenv
load_dotenv()

import ollama
from rich.console import Console
from rich.panel import Panel

console = Console()

# Configuration
OLLAMA_MODEL = "gpt-oss:20b-cloud"

def refine_query(chat_history, latest_query):
    """Rewrites a vague query based on chat history into a standalone query using Ollama."""
    prompt = f"""
    Given the following chat history and a follow-up question, 
    rewrite the follow-up question to be a standalone search query that contains all necessary context.
    
    Chat History:
    {chat_history}
    
    Follow-up Question: {latest_query}
    
    Standalone Query:
    """
    
    response = ollama.generate(model=OLLAMA_MODEL, prompt=prompt)
    return response['response'].strip()

def run_demo():
    # Scenario: A conversation about the Eiffel Tower
    chat_history = """
    User: Tell me about the Eiffel Tower.
    AI: The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It was named after the engineer Gustave Eiffel.
    """
    
    vague_query = "How tall is it?"
    
    console.print(Panel(f"[bold cyan]Chat History:[/bold cyan]\n{chat_history.strip()}"))
    console.print(f"[bold red]User follow-up:[/bold red] {vague_query}")
    
    # Step 1: Refine
    console.print(f"\n[yellow]Refining query using {OLLAMA_MODEL} and history...[/yellow]")
    standalone_query = refine_query(chat_history, vague_query)
    
    console.print(Panel(f"[bold green]Standalone Query for Vector DB:[/bold green]\n{standalone_query}"))

    console.print("\n[bold yellow]Why this is crucial for RAG:[/bold yellow]")
    console.print("1. [bold]Anaphora Resolution:[/bold] Vector databases don't understand 'it', 'him', or 'that project'.")
    console.print("2. [bold]Precision:[/bold] Searching for 'How tall is it?' would give random results about height.")
    console.print("3. [bold]Searchable Context:[/bold] Searching for 'How tall is the Eiffel Tower?' gives exactly what is needed.")

if __name__ == "__main__":
    run_demo()
