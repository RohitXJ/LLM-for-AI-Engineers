import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rich.console import Console
from rich.panel import Panel

console = Console()

# We'll use a fast, lightweight model for this demo
model = SentenceTransformer('all-MiniLM-L6-v2')

TEXT = """
Apple Inc. is an American multinational technology company. 
It is headquartered in Cupertino, California. 
Apple is the world's largest technology company by revenue.

Polar bears are large carnivorous mammals. 
They are native to the Arctic Circle. 
Their white fur helps them blend into the snowy environment.

Quantum computing is a type of computing that uses quantum-mechanical phenomena. 
Superposition and entanglement are key principles. 
A quantum computer can perform certain calculations much faster than classical computers.
"""

def main():
    console.rule("[bold magenta]Semantic Splitting Demo[/bold magenta]")
    
    # 1. Split into sentences (simple version)
    sentences = [s.strip() for s in TEXT.split('\n') if s.strip()]
    console.print(f"Detected {len(sentences)} sentences.\n")

    # 2. Get embeddings for each sentence
    embeddings = model.encode(sentences)

    # 3. Calculate similarity between adjacent sentences
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
        similarities.append(sim)
        
    # 4. Find the 'jumps' (where similarity is low)
    # We'll use a simple threshold or just look for the lowest values
    threshold = 0.5 
    
    current_chunk = [sentences[0]]
    chunks = []
    
    for i in range(len(similarities)):
        sim = similarities[i]
        next_sentence = sentences[i+1]
        
        # If similarity is low, start a new chunk
        if sim < threshold:
            chunks.append(" ".join(current_chunk))
            current_chunk = [next_sentence]
        else:
            current_chunk.append(next_sentence)
            
    chunks.append(" ".join(current_chunk))

    # 5. Visualize
    for i, chunk in enumerate(chunks):
        console.print(Panel(chunk, title=f"Semantic Chunk {i+1}", border_style="yellow"))

    console.print("\n[bold cyan]Senior Insight:[/bold cyan]")
    console.print("Notice how the chunks grouped sentences by 'Topic' (Apple, Polar Bears, Quantum).")
    console.print("Even though they were in the same text block, the semantic distance triggered the split.")

if __name__ == "__main__":
    main()
