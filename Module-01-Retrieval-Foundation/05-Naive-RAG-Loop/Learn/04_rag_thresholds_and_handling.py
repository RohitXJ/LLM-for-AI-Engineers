import os
import ollama
import chromadb
from chromadb.utils import embedding_functions

# Configuration
MODEL = "gpt-oss:20b-cloud"
# Note: Default Chroma distance is squared L2 distance (lower is better)
# For this embedding function, anything above 1.0 is usually irrelevant.
RELEVANCE_THRESHOLD = 1.0 

default_ef = embedding_functions.DefaultEmbeddingFunction()
db_client = chromadb.EphemeralClient()
collection = db_client.get_or_create_collection(
    name="threshold_test",
    embedding_function=default_ef
)

def prepare_db():
    facts = ["The secret password for the vault is 'Blueberry-123'."]
    print("--- Preparing Threshold Database ---")
    collection.add(ids=["secret"], documents=facts)

def smart_retrieval(query):
    # Chroma automatically embeds the string using default_ef
    results = collection.query(query_texts=[query], n_results=1)
    
    # Extract the distance score from the nested list structure [0][0]
    distance = results["distances"][0][0]
    document = results["documents"][0][0]
    
    print(f"--- Query: '{query}' | Distance: {distance:.4f} ---")
    
    if distance > RELEVANCE_THRESHOLD:
        return None  # Too far away, likely irrelevant
    return document

def chat(query):
    context = smart_retrieval(query)
    
    if context:
        prompt = f"Use this context to answer concisely: {context}\n\nQuestion: {query}"
        print("Action: Retrieval Successful. Using Context.")
    else:
        prompt = query
        print("Action: No relevant context found. Asking LLM directly.")

    response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
    print(f"Response: {response['message']['content']}\n")

if __name__ == "__main__":
    prepare_db()
    
    # Case 1: Relevant Query
    chat("What is the vault password?")
    
    # Case 2: Irrelevant Query (should trigger threshold)
    chat("How do I bake a cake?")
