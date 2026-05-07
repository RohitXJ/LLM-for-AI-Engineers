from dotenv import load_dotenv
load_dotenv()

from sentence_transformers import CrossEncoder
from rich.console import Console
from rich.panel import Panel

console = Console()

class LocalReranker:
    """
    An Adapter class that makes a local Cross-Encoder 
    behave like a professional Rerank API.
    """
    def __init__(self, model_name='BAAI/bge-reranker-base'):
        console.print(f"[dim]Initializng Local Reranker: {model_name}...[/dim]")
        self.model = CrossEncoder(model_name)

    def rerank(self, query, documents, top_n=3):
        # 1. Prepare pairs
        pairs = [[query, doc] for doc in documents]
        
        # 2. Get scores
        scores = self.model.predict(pairs)
        
        # 3. Create a list of objects (like Cohere does)
        results = []
        for idx, score in enumerate(scores):
            results.append({
                "index": idx,
                "score": score,
                "text": documents[idx]
            })
            
        # 4. Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # 5. Return only top_n
        return results[:top_n]

# --- USAGE EXAMPLE ---
def main():
    # Now using the reranker is as easy as an API!
    my_reranker = LocalReranker()
    
    query = "What is the best way to handle large scale vector search?"
    docs = [
        "Use a distributed vector database like Qdrant or Milvus.",
        "Store your vectors in a CSV file for simplicity.",
        "Implement HNSW indexing for sub-second search speeds.",
        "Vectors are just arrays of numbers.",
        "Product Quantization (PQ) helps compress large vector collections."
    ]

    # CLEAN SYNTAX: Just like Cohere!
    top_results = my_reranker.rerank(query, docs, top_n=2)

    console.print(f"\n[bold cyan]Query:[/bold cyan] {query}\n")
    console.print("[bold green]Top Reranked Results:[/bold green]")
    
    for r in top_results:
        console.print(Panel(
            f"{r['text']}\n[dim]Index: {r['index']} | Score: {r['score']:.4f}[/dim]",
            expand=False
        ))

if __name__ == "__main__":
    main()
