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

def generate_hypothetical_doc(query):
    """Generates a hypothetical answer/document using Ollama."""
    prompt = f"""
    Please write a scientific/technical paragraph to answer the following question. 
    Even if you don't know the exact answer, provide a detailed and plausible response that would appear in a knowledge base.
    
    Question: {query}
    
    Hypothetical Answer:
    """
    
    response = ollama.generate(model=OLLAMA_MODEL, prompt=prompt)
    return response['response'].strip()

def run_demo():
    # 1. Setup In-Memory ChromaDB
    client = chromadb.EphemeralClient()
    collection = client.create_collection(name="hyde_demo")
    
    knowledge_base = [
        "The XJ-900 logic gate operates at a frequency of 5.4GHz and requires a stable 1.2V supply.",
        "To initialize the quantum-state-buffer, the user must first clear the parity registers.",
        "The standard protocol for deep-sea data transmission involves ultrasonic modulation at 20kHz.",
        "Photosynthetic efficiency in urban vertical farms can be increased by using blue-spectrum LEDs.",
        "Neural network pruning typically involves removing weights with the smallest absolute values.",
        "The capital of Mars is Utopia Planitia in the fictional 2150 timeline.",
        "Baking soda and vinegar react to produce carbon dioxide gas and sodium acetate."
    ]
    
    collection.add(
        documents=knowledge_base,
        ids=[f"id_{i}" for i in range(len(knowledge_base))]
    )
    
    user_query = "What are the power requirements for the XJ-900 gate?"
    
    console.print(Panel(f"[bold cyan]User Query:[/bold cyan] {user_query}"))
    
    # --- METHOD 1: STANDARD RETRIEVAL ---
    console.print("\n[yellow]Performing Standard Retrieval...[/yellow]")
    res_std = collection.query(query_texts=[user_query], n_results=1)
    doc_standard = res_std['documents'][0][0]
    dist_standard = res_std['distances'][0][0]
    
    console.print(f"  Best Match: {doc_standard} [dim](Dist: {dist_standard:.4f})[/dim]")

    # --- METHOD 2: HyDE RETRIEVAL ---
    console.print(f"\n[yellow]Generating Hypothetical Document (HyDE) using {OLLAMA_MODEL}...[/yellow]")
    hypo_doc = generate_hypothetical_doc(user_query)
    console.print(Panel(f"[italic]{hypo_doc}[/italic]", title="Hypothetical Answer", border_style="dim"))
    
    console.print("\n[yellow]Performing HyDE Retrieval (using the hallucinated text)...[/yellow]")
    res_hyde = collection.query(query_texts=[hypo_doc], n_results=1)
    doc_hyde = res_hyde['documents'][0][0]
    dist_hyde = res_hyde['distances'][0][0]
    
    console.print(f"  Best Match: {doc_hyde} [dim](Dist: {dist_hyde:.4f})[/dim]")

    console.print("\n[bold green]Why HyDE works:[/bold green]")
    console.print("1. [bold]Matching Style:[/bold] A question ('What is...') doesn't look like a statement ('The XJ-900...').")
    console.print("2. [bold]Semantic Overlap:[/bold] The LLM fills the hypothetical doc with related keywords (voltage, supply, electricity).")
    console.print("3. [bold]Closer Vector:[/bold] The hypothetical answer's vector is often much closer to the target document's vector.")

if __name__ == "__main__":
    run_demo()
