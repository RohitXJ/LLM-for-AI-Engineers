import chromadb
from chromadb.utils import embedding_functions

# 1. Define a Custom Embedding Function
# Here we are explicitly choosing a model from HuggingFace (Sentence Transformers)
# 'all-mpnet-base-v2' is larger and more accurate than the default 'all-MiniLM-L6-v2'
model_name = "all-mpnet-base-v2"
huggingface_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)

# 2. Initialize Client
client = chromadb.Client()

# 3. Create a collection with a SPECIFIC Distance Metric
# 'hnsw:space' defines the distance metric. Options: "l2", "ip", or "cosine"
print(f"Creating collection with '{model_name}' and 'cosine' similarity...")
collection = client.create_collection(
    name="advanced_metrics_demo",
    embedding_function=huggingface_ef,
    metadata={"hnsw:space": "cosine"} 
)

# 4. Add data
documents = [
    "The concept of gravity explains why objects fall toward the Earth.",
    "Quantum entanglement is a phenomenon where particles become correlated.",
    "Pizza dough is made from flour, water, yeast, and salt.",
    "The capital of France is Paris."
]

collection.add(
    documents=documents,
    ids=[f"id{i}" for i in range(len(documents))]
)

# 5. Semantic Search with Score (Distance)
query = "Explain a scientific law about physical forces"
print(f"\nQuery: '{query}'")

results = collection.query(
    query_texts=[query],
    n_results=2,
    include=["documents", "distances", "metadatas"] # Ask Chroma to return the distance scores!
)

# 6. Display results with 'Confidence'
print("\n--- Results with Distance Scores ---")
for doc, distance in zip(results['documents'][0], results['distances'][0]):
    # Note: In 'cosine' space, distance is 1 - similarity. 
    # Distance 0.0 means identical. Distance 1.0+ means very different.
    print(f"Score: {distance:.4f} | Doc: {doc}")

print("\nEngineering Note: Lower distance = Higher similarity.")
