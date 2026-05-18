import os
from dotenv import load_dotenv
load_dotenv()

import ollama
import chromadb
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# Configuration
OLLAMA_MODEL = "gpt-oss:20b-cloud"

def get_multi_queries(query, n=3):
    """Uses Ollama to generate multiple variations of the query."""
    prompt = f"""
    You are an AI assistant helping to improve search retrieval.
    The user asked: "{query}"
    
    Generate {n} different variations of this question that capture the same intent but use different wording or perspective.
    Focus on synonyms and different sentence structures.
    
    Output only the queries, one per line, no numbering.
    """
    
    response = ollama.generate(model=OLLAMA_MODEL, prompt=prompt)
    queries = [q.strip() for q in response['response'].strip().split('\n') if q.strip()]
    return queries[:n]

def run_demo():
    # 1. Setup In-Memory ChromaDB
    client = chromadb.EphemeralClient()
    collection = client.create_collection(name="multi_query_demo")
    
    knowledge_base = [
        "The project deadline is scheduled for next Friday at 5 PM.",
        "Team members must submit their progress reports by end of week.",
        "The final presentation will be held in the main conference room.",
        "All code must be pushed to the 'main' branch before the release.",
        "The software architecture uses a microservices pattern with Docker.",
        "Documentation is available on the internal wiki page.",
        "Contact Sarah from HR for any leave requests.",
        "The annual company retreat is planned for July in the Alps.",
        "Security patches should be applied immediately to all production servers.",
        "The budget for Q3 has been approved by the board."
    ]
    
    # Add to collection (Chroma handles embedding with default model)
    collection.add(
        documents=knowledge_base,
        ids=[f"id_{i}" for i in range(len(knowledge_base))]
    )
    
    user_query = "When do I need to finish the work?"
    
    console.print(Panel(f"[bold cyan]Original Query:[/bold cyan] {user_query}"))
    
    # Step 1: Generate Multi-Queries
    console.print(f"\n[yellow]Generating Multi-Query variations using {OLLAMA_MODEL}...[/yellow]")
    multi_queries = get_multi_queries(user_query, n=3)
    all_queries = [user_query] + multi_queries
    
    for i, q in enumerate(all_queries):
        console.print(f"  {i}. {q}")
        
    # Step 2: Retrieve for EACH query
    console.print("\n[yellow]Retrieving from In-Memory ChromaDB for all variations...[/yellow]")
    
    unique_results = {} # Use dict to store doc: min_distance (Chroma uses L2 by default, smaller is better)
    
    for q in all_queries:
        results = collection.query(query_texts=[q], n_results=2)
        
        for doc, dist in zip(results['documents'][0], results['distances'][0]):
            # If doc already found, keep the one with smaller distance
            if doc not in unique_results or dist < unique_results[doc]:
                unique_results[doc] = dist

    # Step 3: Sort and Display
    # Sort by distance (ascending)
    sorted_results = sorted(unique_results.items(), key=lambda x: x[1])
    
    table = Table(title="Multi-Query Retrieval Results (In-Memory Chroma)")
    table.add_column("Rank", justify="right", style="cyan")
    table.add_column("Document", style="white")
    table.add_column("Distance (L2)", justify="right", style="green")
    
    for i, (doc, dist) in enumerate(sorted_results[:5]):
        table.add_row(str(i+1), doc, f"{dist:.4f}")
        
    console.print(table)
    
    console.print("\n[bold green]Theory Check:[/bold green]")
    console.print("1. [bold]Diversity:[/bold] The original query was vague ('finish the work').")
    console.print("2. [bold]Expansion:[/bold] Variations like 'deadline' or 'submit reports' likely hit different documents.")
    console.print("3. [bold]Recall:[/bold] By combining results, we ensure we don't miss relevant info due to phrasing.")

if __name__ == "__main__":
    run_demo()
