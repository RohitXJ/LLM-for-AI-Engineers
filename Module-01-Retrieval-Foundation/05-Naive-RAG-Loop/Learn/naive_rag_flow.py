import os
# Suppress all Hugging Face / SBERT noise
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import logging
from transformers.utils import logging as tf_logging
tf_logging.set_verbosity_error()
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)




import ollama
import chromadb
from sentence_transformers import SentenceTransformer

# Configuration
MODEL = "qwen3.5:0.8b"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2" # Tiny, fast, and high quality
COLLECTION_NAME = "naive_rag_local_sbert"

# Initialize local Embedding Model (Sentence Transformers)
embedder = SentenceTransformer(EMBED_MODEL_NAME)

# Initialize local Vector Storage (In-Memory)
db_client = chromadb.EphemeralClient()
collection = db_client.get_or_create_collection(name=COLLECTION_NAME)

def ingest_data(data_points):
    """Embeds text into vectors using SBERT and stores them in ChromaDB."""
    print(f"--- Ingesting {len(data_points)} facts ---")
    for i, text in enumerate(data_points):
        # Generate embedding locally via Sentence Transformers
        embedding = embedder.encode(text).tolist()
        
        collection.add(
            ids=[f"id_{i}"],
            embeddings=[embedding],
            documents=[text]
        )
    print("Ingestion Complete.\n")

def retrieve_context(query, n_results=2):
    """Retrieves the most relevant text chunks for a query."""
    # Embed the query
    query_embed = embedder.encode(query).tolist()
    
    results = collection.query(
        query_embeddings=[query_embed],
        n_results=n_results
    )
    return "\n".join(results["documents"][0])

def generate_answer(query, context):
    """Augments the prompt with context and streams the LLM response."""
    system_prompt = (
        "You are a Senior AI Engineer. Answer using ONLY the provided context. "
        "If unknown, say 'Information not found in database.'"
    )
    user_prompt = f"CONTEXT:\n{context}\n\nQUESTION: {query}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    for part in ollama.chat(model=MODEL, messages=messages, stream=True):
        print(part['message']['content'], end='', flush=True)

if __name__ == "__main__":
    knowledge_base = [
        "The project 'LLM-for-AI-Engineers' is a master blueprint for building production-grade AI systems.",
        "Module 01 covers Retrieval Foundations.",
        "Naive RAG consists of: Ingestion, Retrieval, and Generation.",
        "ChromaDB serves as the local vector database."
    ]

    ingest_data(knowledge_base)
    
    query = "What are the steps of Naive RAG?"
    context = retrieve_context(query)
    
    print(f"Query: {query}\nResponse: ", end="")
    generate_answer(query, context)
