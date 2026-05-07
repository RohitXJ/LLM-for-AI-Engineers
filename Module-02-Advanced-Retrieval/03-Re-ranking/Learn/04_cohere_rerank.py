import os
from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.table import Table

console = Console()

def cohere_rerank_demo():
    api_key = os.getenv("COHERE_API_KEY")
    
    query = "How do I optimize my vector database for speed?"
    docs = [
        "To speed up searches, use HNSW indexing in your vector DB.",
        "Vector databases store embeddings as high-dimensional points.",
        "Quantization reduces the memory footprint of your vectors.",
        "Python is a popular language for AI engineering.",
        "Increasing the 'probes' in IVF indexing improves accuracy but slows down speed.",
        "A fast database is essential for a good user experience."
    ]

    console.print(f"\n[bold cyan]Query:[/bold cyan] {query}\n")

    if not api_key or api_key == "YOUR_COHERE_API_KEY":
        console.print("[bold red]No Cohere API Key found in .env[/bold red]")
        console.print("Showing you the code pattern instead...\n")
        
        # This is the standard pattern you'd use:
        code_example = """
        import cohere
        co = cohere.Client(api_key)
        
        response = co.rerank(
            model='rerank-v3.5',
            query=query,
            documents=docs,
            top_n=3
        )
        
        for result in response.results:
            print(f"Document Index: {result.index}")
            print(f"Relevance Score: {result.relevance_score}")
            print(f"Text: {docs[result.index]}")
        """
        console.print(code_example, style="dim")
        return

    # If key exists, run the actual API call
    try:
        import cohere
        co = cohere.Client(api_key)
        
        console.print("[yellow]Calling Cohere Rerank API...[/yellow]")
        response = co.rerank(
            model='rerank-v3.5',
            query=query,
            documents=docs,
            top_n=3
        )

        table = Table(title="Cohere Rerank Results")
        table.add_column("Rank", style="magenta")
        table.add_column("Score", style="green")
        table.add_column("Document", style="white")

        for i, result in enumerate(response.results):
            table.add_row(
                str(i+1),
                f"{result.relevance_score:.4f}",
                docs[result.index]
            )
        
        console.print(table)

    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")

if __name__ == "__main__":
    cohere_rerank_demo()
