from dotenv import load_dotenv
load_dotenv()

from sentence_transformers import SentenceTransformer, CrossEncoder, util
import torch
from rich.console import Console
from rich.panel import Panel

console = Console()

def run_pipeline():
    # 1. Setup Models
    # Fast retriever (Bi-Encoder)
    retriever = SentenceTransformer('all-MiniLM-L6-v2')
    # Accurate Re-ranker (Cross-Encoder)
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    # 2. Our "Database" of facts
    knowledge_base = [
        "The capital of France is Paris.",
        "The Eiffel Tower was completed in 1889.",
        "Paris is known for its cafe culture and the Louvre museum.",
        "Berlin is the capital of Germany and is famous for its history.",
        "The Great Wall of China is a series of fortifications.",
        "London is the capital of England.",
        "The Louvre in Paris is the world's largest art museum.",
        "French cuisine is world-renowned for its sophistication.",
        "The population of Paris is over 2 million people.",
        "Mount Everest is the highest mountain in the world."
    ]

    query = "Tell me about the famous museum in the French capital."

    # --- STEP 1: RETRIEVAL (The Bi-Encoder Phase) ---
    # In a real app, this would be a ChromaDB query.
    console.print(f"[bold yellow]1. Retrieving top 5 from {len(knowledge_base)} docs using Bi-Encoder...[/bold yellow]")
    
    query_emb = retriever.encode(query, convert_to_tensor=True)
    kb_embs = retriever.encode(knowledge_base, convert_to_tensor=True)
    
    hits = util.semantic_search(query_emb, kb_embs, top_k=5)[0]
    
    # Get the actual text for the hits
    retrieved_docs = [knowledge_base[hit['corpus_id']] for hit in hits]
    
    console.print("Top 5 Retrieved Docs:")
    for i, doc in enumerate(retrieved_docs):
        console.print(f"  {i+1}. {doc}")

    # --- STEP 2: RE-RANKING (The Cross-Encoder Phase) ---
    console.print(f"\n[bold green]2. Re-ranking those 5 docs using Cross-Encoder...[/bold green]")
    
    # Prepare pairs: [query, doc]
    pairs = [[query, doc] for doc in retrieved_docs]
    
    # Predict relevance
    rerank_scores = reranker.predict(pairs)
    
    # Combine docs with their new scores
    reranked_results = list(zip(retrieved_docs, rerank_scores))
    
    # Sort by score in descending order
    reranked_results.sort(key=lambda x: x[1], reverse=True)

    # --- DISPLAY FINAL RESULTS ---
    console.print("\n[bold cyan]Final Re-ranked Result (Top 1):[/bold cyan]")
    final_doc, final_score = reranked_results[0]
    console.print(Panel(f"{final_doc}\n[dim]Re-rank Score: {final_score:.4f}[/dim]", expand=False))

    console.print("\n[bold yellow]Observation:[/bold yellow]")
    console.print("The Bi-Encoder might have retrieved several 'Paris' or 'Museum' docs.")
    console.print("The Re-ranker carefully weighed the query tokens 'famous museum' AND 'French capital'")
    console.print("to find the specific doc that mentions the Louvre in Paris.")

if __name__ == "__main__":
    run_pipeline()
