import chromadb
import os
import shutil

# 1. Setup Persistent Storage
# This will create a folder named 'chroma_storage' in your current directory
db_path = "chroma_storage"

# Cleanup from previous runs if any
if os.path.exists(db_path):
    print(f"Cleaning up existing DB at {db_path}...")
    # In a real app, you wouldn't delete this! You'd just connect to it.
    shutil.rmtree(db_path)

# Initialize Persistent Client
client = chromadb.PersistentClient(path=db_path)

# 2. Create/Get Collection
collection = client.create_collection(name="engineering_docs")

# 3. Add Documents with rich metadata
print("Adding documents with metadata...")
collection.add(
    documents=[
        "Building a scalable API with FastAPI and Python.",
        "How to optimize Postgres queries for high traffic.",
        "Intro to Kubernetes: Orchestrating containers at scale.",
        "Python design patterns for AI Engineers.",
        "Advanced CSS techniques for modern web apps."
    ],
    metadatas=[
        {"tags": "backend", "difficulty": "intermediate", "year": 2023},
        {"tags": "database", "difficulty": "advanced", "year": 2022},
        {"tags": "devops", "difficulty": "advanced", "year": 2024},
        {"tags": "backend", "difficulty": "beginner", "year": 2024},
        {"tags": "frontend", "difficulty": "intermediate", "year": 2023}
    ],
    ids=["doc1", "doc2", "doc3", "doc4", "doc5"]
)

# 4. Filtered Query
# We want to find "Python" stuff, but ONLY if it is marked as 'backend'
print("\n--- Running Filtered Query ---")
results = collection.query(
    query_texts=["Tell me about Python development"],
    n_results=2,
    where={"tags": "backend"} # This is the Metadata Filter!
)

for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
    print(f"Result: {doc} | Metadata: {meta}")

# 5. The 'U' in CRUD (Update)
print("\n--- Updating 'doc1' content ---")
collection.update(
    ids=["doc1"],
    documents=["Building a scalable API with FastAPI and Python (v2 - Updated)."],
    metadatas=[{"tags": "backend", "difficulty": "advanced", "year": 2024}]
)

# Verify Update
updated_res = collection.get(ids=["doc1"])
print(f"Verified Update: {updated_res['documents'][0]}")

print(f"\nSuccess! Your data is now saved in the '{db_path}' folder.")
