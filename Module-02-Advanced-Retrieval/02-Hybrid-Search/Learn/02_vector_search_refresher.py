import chromadb
from sentence_transformers import SentenceTransformer
from typing import List

# 1. Setup local embedding model
# This model converts text into 384-dimensional vectors
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Setup ChromaDB (In-memory for this demo)
client = chromadb.Client()
collection = client.create_collection(name="refresher_collection")

# 3. Our Knowledge Base
documents = [
    "The iPhone 15 Pro Max has a titanium frame and a high-end camera system.",
    "Samsung Galaxy S24 Ultra features an AI-integrated S-Pen and incredible zoom.",
    "MacBook Pro with M3 Max is the ultimate laptop for creative professionals.",
    "The new Sony headphones have industry-leading noise cancellation technology.",
    "How to fix a 'Connection Timeout' error in your Python database script."
]

# 4. Ingest data
# Note: Chroma normally handles embeddings automatically if you provide a model, 
# but we'll do it manually to see the logic.
embeddings = model.encode(documents).tolist()

collection.add(
    ids=[f"doc_{i}" for i in range(len(documents))],
    embeddings=embeddings,
    documents=documents
)

def search_vector(query: str, top_k: int = 2):
    print(f"\n🌐 Vector Searching for: '{query}'")
    query_embedding = model.encode([query]).tolist()
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    
    for i in range(len(results['documents'][0])):
        doc = results['documents'][0][i]
        dist = results['distances'][0][i]
        print(f"Distance: {dist:.4f} | Content: {doc[:50]}...")
    
    return results

if __name__ == "__main__":
    # Test 1: Semantic Match (This WILL work!)
    # Even though 'smartphone' isn't in the text, it knows iPhone/Samsung are smartphones.
    search_vector("high end smartphone")
    
    # Test 2: Specific Acronym/Term (Might be less precise than BM25)
    search_vector("S24 Ultra")
    
    # Test 3: Error codes (Vector models sometimes struggle with specific code numbers)
    search_vector("Error 404 connection")
