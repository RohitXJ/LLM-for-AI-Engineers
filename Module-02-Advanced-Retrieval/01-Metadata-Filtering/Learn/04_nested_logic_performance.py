import chromadb

# -----------------------------------------------------
# COMPLEX NESTED LOGIC
# -----------------------------------------------------
# ChromaDB supports deeply nested $and and $or.
# Example: "Find documents that are (Security AND High Priority) OR (HR AND 2024)"

client = chromadb.Client() # Memory client for quick demo
collection = client.create_collection("nested_demo")

# Add some dummy data
collection.add(
    ids=["1", "2", "3", "4"],
    documents=["Doc A", "Doc B", "Doc C", "Doc D"],
    metadatas=[
        {"dept": "Security", "priority": "High", "year": 2023},
        {"dept": "HR", "priority": "Low", "year": 2024},
        {"dept": "Security", "priority": "Low", "year": 2024},
        {"dept": "IT", "priority": "High", "year": 2024}
    ]
)

# NESTED FILTER: (Security AND High) OR (HR AND 2024)
nested_filter = {
    "$or": [
        {
            "$and": [
                {"dept": "Security"},
                {"priority": "High"}
            ]
        },
        {
            "$and": [
                {"dept": "HR"},
                {"year": 2024}
            ]
        }
    ]
}

print("\n--- Running Nested Logic Query ---")
results = collection.query(
    query_texts=["Anything"],
    where=nested_filter,
    n_results=10
)

for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"-> Found: {doc} | Metadata: {meta}")
