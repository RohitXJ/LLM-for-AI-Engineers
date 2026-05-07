from dotenv import load_dotenv
load_dotenv()

from sentence_transformers import SentenceTransformer, CrossEncoder, util
import torch
from rich.console import Console
from rich.table import Table

console = Console()

def demonstrate_reranking():
    # 1. Load Models
    # Bi-Encoder: Fast, maps sentences to a vector space.
    # Good for: Searching millions of docs.
    bi_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Cross-Encoder: Slow, processes Query + Doc together.
    # Good for: High-precision ranking of a small subset (e.g., top 25).
    cross_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    query = "How can I get a refund for my subscription?"

    # Tricky documents that might confuse a simple vector search
    docs = [
        "Our refund policy allows for returns within 30 days of purchase.",
        "To cancel your subscription, go to settings and click deactivate.",
        "I love my new subscription, it's the best purchase I've made!",
        "Refunds are not available for monthly plans, only annual ones.",
        "The subscription model is changing next year for all users."
    ]

    console.print(f"\n[bold cyan]Query:[/bold cyan] {query}\n")

    # --- Stage 1: Bi-Encoder (Vector Similarity) ---
    query_emb = bi_model.encode(query, convert_to_tensor=True)
    doc_embs = bi_model.encode(docs, convert_to_tensor=True)
    
    # Compute Cosine Similarity
    cos_scores = util.cos_sim(query_emb, doc_embs)[0]
    
    # --- Stage 2: Cross-Encoder (Re-ranking) ---
    # We pair the query with each doc: [[query, doc1], [query, doc2], ...]
    pairs = [[query, doc] for doc in docs]
    cross_scores = cross_model.predict(pairs)

    # --- Display Results ---
    table = Table(title="Bi-Encoder vs. Cross-Encoder Scores")
    table.add_column("Document Content", style="white", no_wrap=False)
    table.add_column("Bi-Encoder (Sim)", style="yellow", justify="center")
    table.add_column("Cross-Encoder (Score)", style="green", justify="center")

    for i in range(len(docs)):
        table.add_row(
            docs[i], 
            f"{cos_scores[i]:.4f}", 
            f"{cross_scores[i]:.4f}"
        )

    console.print(table)
    
    console.print("\n[bold yellow]Analysis:[/bold yellow]")
    console.print("1. Notice how the Bi-Encoder gives high scores to docs with shared keywords like 'subscription'.")
    console.print("2. The Cross-Encoder (Score) is often on a different scale (can be negative).")
    console.print("3. Look at the doc: 'I love my new subscription...'. The Bi-Encoder might rank it high due to 'subscription', but the Cross-Encoder should see it's irrelevant to 'refund'.")

if __name__ == "__main__":
    demonstrate_reranking()
