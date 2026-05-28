from shared_llm.llm import ChatManager
from shared_llm.database import ChromaManager
from shared_llm.processing import DataLoader, Chunker
from shared_llm.reranking import LocalReranker
from shared_llm.retrieval import KeywordEngine, HybridFusion
import os
from pathlib import Path

DATA_PATH = "data/"

def main():
    # 1. Load context from data directory
    context = DataLoader.master_loader(DATA_PATH, allowed_files=[".txt", ".md", ".json"])
    
    # 2. Initialize Chroma and Chunker
    chroma = ChromaManager("./chroma")
    chunker = Chunker()
    
    # 3. Identify which files are already in the database
    # We use the 'source' metadata to track uniqueness
    existing_sources = chroma.get_unique_metadata_values("source")
    
    # 4. Filter for only new documents
    new_docs = [doc for doc in context if doc["metadata"]["source"] not in existing_sources]
    
    if new_docs:
        print(f"\nNew data found: {list(set([d['metadata']['source'] for d in new_docs]))}. Ingesting...")
        chroma.add_documents(new_docs, chunker=chunker)
        print("Ingestion complete.")
    else:
        print("\nNo new data to ingest.")

if __name__ == "__main__":
    main()
