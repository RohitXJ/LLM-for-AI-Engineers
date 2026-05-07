from dotenv import load_dotenv
import os

# Load environment variables (HF_HOME, etc.) from .env file
load_dotenv()

import time
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from rich.console import Console
from rich.table import Table

console = Console()

def measure_latency():
    # 1. Initialize Models
    console.print("[bold yellow]Loading models...[/bold yellow]")
    retriever = SentenceTransformer('all-MiniLM-L6-v2')
    
    # BGE-Reranker is one of the most powerful open-source rerankers
    # We use the 'v2-m3' or 'base' version for a balance of speed and power
    reranker = CrossEncoder('BAAI/bge-reranker-base')

    # 2. Setup a larger dummy dataset
    # Imagine we retrieved 50 documents from a database
    query = "What are the health benefits of Mediterranean diet?"
    
    # Simulating 50 retrieved documents
    retrieved_docs = [
        "The Mediterranean diet emphasizes fruits and vegetables.",
        "Paris is a beautiful city in France.",
        "Olive oil is a key component of the Mediterranean diet.",
        "Regular exercise is important for heart health.",
        "Studies show the Mediterranean diet reduces heart disease risk.",
        "The sun rises in the east every morning.",
        "Wine in moderation is often included in Mediterranean meals.",
        "How to bake a chocolate cake at home.",
        "Fish and poultry are preferred over red meat in this diet.",
        "The stock market saw a slight dip today."
    ] * 5  # Creating 50 docs total

    console.print(f"\n[bold cyan]Query:[/bold cyan] {query}")
    console.print(f"Number of docs to process: {len(retrieved_docs)}\n")

    # --- MEASURE RETRIEVAL SPEED (SIMULATED) ---
    start_time = time.time()
    query_emb = retriever.encode(query)
    doc_embs = retriever.encode(retrieved_docs)
    retrieval_time = time.time() - start_time
    
    # --- MEASURE RE-RANKING SPEED ---
    start_time = time.time()
    pairs = [[query, doc] for doc in retrieved_docs]
    scores = reranker.predict(pairs)
    rerank_time = time.time() - start_time

    # --- DISPLAY METRICS ---
    table = Table(title="Latency Comparison (50 Documents)")
    table.add_column("Phase", style="magenta")
    table.add_column("Time (Seconds)", style="green")
    table.add_column("Notes", style="white")

    table.add_row("Bi-Encoder (Encoding)", f"{retrieval_time:.4f}s", "Fast, but only creates vectors")
    table.add_row("Cross-Encoder (Full Rerank)", f"{rerank_time:.4f}s", "Slow, compares Query to every Doc")
    
    console.print(table)

    # --- THE SENIOR ENGINEER'S RULE ---
    console.print("\n[bold red]The 'Top-K' Rule:[/bold red]")
    if rerank_time > 0.5:
        console.print("⚠️ Re-ranking 50 docs is taking too long for a real-time chatbot!")
        console.print("💡 [bold green]Optimization:[/bold green] Only re-rank the top 10-20 docs from the retriever.")
    else:
        console.print("✅ Speed is acceptable for this small batch.")

if __name__ == "__main__":
    measure_latency()
