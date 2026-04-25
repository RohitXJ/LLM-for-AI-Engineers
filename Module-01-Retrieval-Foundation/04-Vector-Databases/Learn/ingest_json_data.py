import chromadb
import json
import os

# 1. Configuration
JSON_PATH = os.path.join("Module-01-Retrieval-Foundation", "04-Vector-Databases", "Learn", "test_data", "processed_chunks.json")
DB_PATH = "chroma_storage_ingested"

# 2. Load the JSON data
if not os.path.exists(JSON_PATH):
    print(f"Error: Could not find {JSON_PATH}")
    exit()

with open(JSON_PATH, 'r', encoding='utf-8') as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks from JSON.")

# 3. Initialize Chroma
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(name="processed_documents")

# 4. Prepare data for Chroma
# We need three lists: ids, documents, and metadatas
ids = []
documents = []
metadatas = []

for chunk in chunks:
    ids.append(chunk['chunk_id'])
    documents.append(chunk['content'])
    
    # Senior Note: Ensure metadata values are compatible (strings, ints, floats, bools)
    # Our current metadata looks clean, but we pass it as-is.
    metadatas.append(chunk['metadata'])

# 5. Bulk Add
print(f"Ingesting {len(ids)} documents into Chroma...")
collection.add(
    ids=ids,
    documents=documents,
    metadatas=metadatas
)

print("Ingestion complete!")

# 6. Verify with a specific search
# Let's search for the "EMEA login bug" mentioned in the logs
query = "What is VLSI?"
print(f"\nVerifying with query: '{query}'")

results = collection.query(
    query_texts=[query],
    n_results=2
)

print("\n--- Top Retrieved Chunks ---")
for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
    print(f"\nResult {i+1} (Source: {meta['source']}):")
    print(f"Content: {doc[:200]}...") # Print first 200 chars
