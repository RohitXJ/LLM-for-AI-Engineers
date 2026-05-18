import os
from dotenv import load_dotenv
load_dotenv()

import ollama
import chromadb
from rich.console import Console
from rich.panel import Panel

console = Console()

# Configuration
OLLAMA_MODEL = "gpt-oss:20b-cloud"

def decompose_query(query):
    """Decomposes a complex query using Ollama."""
    prompt = f"""
    You are an AI assistant helping to break down complex queries for better search retrieval.
    The user asked: "{query}"
    
    If the question is multi-part or complex, break it down into 2-3 simpler, independent sub-questions.
    If it's already simple, just output the original question.
    
    Output only the sub-questions, one per line, no numbering.
    """
    
    response = ollama.generate(model=OLLAMA_MODEL, prompt=prompt)
    sub_queries = [q.strip() for q in response['response'].strip().split('\n') if q.strip()]
    return sub_queries

def run_demo():
    # 1. Setup In-Memory ChromaDB
    client = chromadb.EphemeralClient()
    collection = client.create_collection(name="decomposition_demo")
    
    knowledge_base = [
        "The project budget is $50,000 for the first phase.",
        "The first phase of the project starts on June 1st.",
        "Team Lead for the project is John Doe.",
        "The second phase involves user testing and feedback loops.",
        "The total duration for all phases is estimated at 6 months.",
        "The tech stack includes Python, React, and PostgreSQL.",
        "John Doe has 10 years of experience in project management."
    ]
    
    collection.add(
        documents=knowledge_base,
        ids=[f"id_{i}" for i in range(len(knowledge_base))]
    )
    
    complex_query = "What is the budget and who is leading the project?"
    
    console.print(Panel(f"[bold cyan]Complex Query:[/bold cyan] {complex_query}"))
    
    # Step 1: Decompose
    console.print(f"\n[yellow]Decomposing into sub-queries using {OLLAMA_MODEL}...[/yellow]")
    sub_queries = decompose_query(complex_query)
    for i, sq in enumerate(sub_queries):
        console.print(f"  {i+1}. {sq}")
        
    # Step 2: Retrieve for each
    console.print("\n[yellow]Retrieving from In-Memory ChromaDB for each sub-query...[/yellow]")
    
    collected_docs = set()
    
    for sq in sub_queries:
        res = collection.query(query_texts=[sq], n_results=1)
        for doc in res['documents'][0]:
            collected_docs.add(doc)
            console.print(f"  [dim]-> Found for '{sq}':[/dim] {doc}")

    # Step 3: Final Results
    console.print("\n[bold green]Final Context Set:[/bold green]")
    for doc in collected_docs:
        console.print(f" - {doc}")

    console.print("\n[bold yellow]Observation:[/bold yellow]")
    console.print("A single search for 'budget and leader' might have prioritized one over the other.")
    console.print("By decomposing, we guaranteed that both 'budget' and 'leader' facts were retrieved.")

if __name__ == "__main__":
    run_demo()
