import chromadb
from chromadb.utils import embedding_functions

# 1. Initialize the Chroma Client
# In-memory mode is perfect for testing (data wipes when the script ends)
client = chromadb.Client()

# 2. Create a Collection
# Think of a Collection like a 'Table' in SQL or a 'Folder' for your vectors
collection = client.create_collection(name="my_first_collection")

# 3. Add Documents
# Chroma handles the embedding for you by default using 'all-MiniLM-L6-v2' (local model)
# You can also provide your own embeddings from OpenAI, HuggingFace, etc.
print("Adding documents to the collection...")
collection.add(
    documents=[
        "The cat is sleeping on the mat.",
        "A robotic arm is assembling a car.",
        "Quantum computing is a type of computation that harnesses collective properties of quantum states.",
        "The pizza was delicious and had extra cheese.",
        "Artificial Intelligence is transforming the tech industry."
    ],
    metadatas=[
        {"category": "animals"},
        {"category": "robotics"},
        {"category": "physics"},
        {"category": "food"},
        {"category": "tech"}
    ],
    ids=["id1", "id2", "id3", "id4", "id5"]
)

# 4. Query the Collection (Semantic Search)
# Note that we search for 'feline', which doesn't exist in our documents!
query_text = "Are there any animals mentioned?"
print(f"\nQuery: '{query_text}'")

results = collection.query(
    query_texts=[query_text],
    n_results=2 # Give me the top 2 matches
)

# 5. Display Results
print("\n--- Search Results ---")
for i in range(len(results['documents'][0])):
    doc = results['documents'][0][i]
    metadata = results['metadatas'][0][i]
    distance = results['distances'][0][i]
    print(f"Match {i+1}: {doc}")
    print(f"   Metadata: {metadata}")
    print(f"   Distance: {distance:.4f} (Lower is more similar)")
