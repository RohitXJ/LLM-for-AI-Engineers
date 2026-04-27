import os
import ollama
import chromadb
from chromadb.utils import embedding_functions

# Configuration
PERSIST_PATH = "./chroma_db_storage"
MODEL = "gpt-oss:20b-cloud"

# 1. Initialize Chroma with Default Embedding Function
# This handles the vectorization automatically
default_ef = embedding_functions.DefaultEmbeddingFunction()
db_client = chromadb.PersistentClient(path=PERSIST_PATH)

# Pass the embedding_function to the collection
collection = db_client.get_or_create_collection(
    name="knowledge_with_sources",
    embedding_function=default_ef
)

def ingest_with_metadata():
    """Ingests data. Chroma handles the embedding automatically."""
    documents = [
        "The AI Engineer Handbook states that RAG is 80% data preparation.",
        "Internal Memo: The company holiday is now moved to December 32nd (Joke).",
        "Project X uses a custom vector engine called 'FastRetriever'."
    ]
    metadatas = [
        {"source": "Handbook_v1", "page": 12},
        {"source": "Internal_Slack", "author": "HR"},
        {"source": "Architecture_Doc", "priority": "high"}
    ]
    ids = [f"id_{i}" for i in range(len(documents))]
    
    if collection.count() == 0:
        print("--- Ingesting data ---")
        # Just pass the documents; Chroma uses default_ef to embed them
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        print("Ingestion successful.")
    else:
        print(f"Database already contains {collection.count()} items.")

def retrieve_with_sources(query):
    """Querying is simpler: just pass the string text."""
    results = collection.query(
        query_texts=[query], # Use query_texts instead of query_embeddings
        n_results=2
    )
    
    # Unpack from the batch structure [0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    
    context_parts = []
    for doc, meta in zip(docs, metas):
        source_info = f"[Source: {meta.get('source')}]"
        context_parts.append(f"{source_info} {doc}")
    
    return "\n".join(context_parts)

def chat_with_citations(query):
    context = retrieve_with_sources(query)
    
    system_prompt = (
        "You are an assistant that MUST cite sources from the provided context. "
        "Answer immediately and concisely."
    )
    user_prompt = f"CONTEXT:\n{context}\n\nQUESTION: {query}"
    
    print(f"\nUser Query: {query}")
    print("-" * 30)
    
    response = ollama.chat(model=MODEL, messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])
    print(response['message']['content'])

if __name__ == "__main__":
    ingest_with_metadata()
    chat_with_citations("When is the company holiday?")
    chat_with_citations("What is Project X using for retrieval?")
