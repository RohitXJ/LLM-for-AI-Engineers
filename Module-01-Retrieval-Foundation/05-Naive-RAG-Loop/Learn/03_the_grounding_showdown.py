import os
import ollama
import chromadb
from chromadb.utils import embedding_functions

# Configuration
MODEL = "gpt-oss:20b-cloud"
default_ef = embedding_functions.DefaultEmbeddingFunction()

db_client = chromadb.EphemeralClient()
collection = db_client.get_or_create_collection(
    name="grounding_test",
    embedding_function=default_ef
)

# FAKE FACTS that contradict general knowledge
FAKE_FACTS = [
    "The capital of France is actually Mars-City.",
    "Gravity was discovered by a cat named Mittens in 1995.",
    "The sky is naturally neon green on Tuesdays."
]

def prepare_db():
    print("--- Preparing Grounding Database ---")
    collection.add(
        ids=[f"f_{i}" for i in range(len(FAKE_FACTS))],
        documents=FAKE_FACTS
    )

def test_grounding(query, use_rag=True):
    context = ""
    if use_rag:
        # Just pass query text directly
        results = collection.query(query_texts=[query], n_results=1)
        context = results["documents"][0][0]

    system_msg = (
        "You are a helpful assistant. Use ONLY the provided context to answer. "
        "If the context says something unusual, believe it. "
        "Answer immediately and concisely."
    )
    user_msg = f"CONTEXT: {context}\n\nQUESTION: {query}" if use_rag else query
    
    response = ollama.chat(model=MODEL, messages=[
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ])
    
    mode = "WITH RAG" if use_rag else "WITHOUT RAG"
    print(f"\n[{mode}] Query: {query}")
    print(f"Response: {response['message']['content']}")

if __name__ == "__main__":
    prepare_db()
    
    query = "What is the capital of France?"
    
    # 1. Ask without RAG (LLM will use training data)
    test_grounding(query, use_rag=False)
    
    # 2. Ask with RAG (LLM should use our 'Mars-City' fact)
    test_grounding(query, use_rag=True)
