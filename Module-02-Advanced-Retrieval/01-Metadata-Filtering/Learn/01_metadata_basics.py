import chromadb
from langchain_huggingface import HuggingFaceEmbeddings

# -----------------------------------------------------
# 1. SETUP & DATA PREPARATION
# -----------------------------------------------------
# Using a persistent client to save data locally
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="metadata_demo")

# Shared Embedding Model
embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Sample data representing a corporate knowledge base
documents = [
    "Security protocol for the main office: Always lock the doors at 6 PM.",
    "Marketing strategy for Q4: Focus on social media engagement.",
    "Security protocol for the satellite office: Cameras are monitored 24/7.",
    "Marketing budget for Q1: Allocation increased by 15%.",
    "Employee handbook: Leave policy and benefits overview."
]

metadatas = [
    {"dept": "Security", "office": "Main", "year": 2023, "priority": "High"},
    {"dept": "Marketing", "office": "Remote", "year": 2023, "priority": "Medium"},
    {"dept": "Security", "office": "Satellite", "year": 2024, "priority": "High"},
    {"dept": "Marketing", "office": "Remote", "year": 2024, "priority": "Low"},
    {"dept": "HR", "office": "Main", "year": 2023, "priority": "Medium"}
]

ids = [f"id_{i}" for i in range(len(documents))]

print("--> Adding documents with rich metadata...")
collection.add(
    ids=ids,
    documents=documents,
    metadatas=metadatas
)

# -----------------------------------------------------
# 2. THE POWER OF FILTERING
# -----------------------------------------------------

def run_query(query_text, filter_dict=None):
    print(f"\n[Query]: '{query_text}'")
    print(f"[Filter]: {filter_dict}")
    
    results = collection.query(
        query_texts=[query_text],
        n_results=2,
        where=filter_dict  # This is where the magic happens
    )
    
    if not results["documents"][0]:
        print("AI-> No documents found matching those filters.")
        return

    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        print(f"AI-> [{meta['dept']} | {meta['office']}] {doc} (Dist: {dist:.4f})")

# --- Scenario A: Simple Equality ---
# Even though "office" is mentioned in many docs, we only want Marketing.
run_query("What is the strategy?", {"dept": "Marketing"})

# --- Scenario B: Logical Operators ($and, $or) ---
# Find security protocols, but ONLY for the satellite office.
run_query("Security protocols", {
    "$and": [
        {"dept": "Security"},
        {"office": "Satellite"}
    ]
})

# --- Scenario C: Comparative Operators ($gt, $gte, $lt, $lte) ---
# Find documents created in 2024.
run_query("Latest updates", {"year": {"$gte": 2024}})

# --- Scenario D: Set Membership ($in, $nin) ---
# Query across multiple departments.
run_query("Important policies", {"dept": {"$in": ["Security", "HR"]}})
