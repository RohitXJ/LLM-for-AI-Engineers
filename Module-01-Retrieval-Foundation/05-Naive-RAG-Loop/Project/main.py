import argparse
import os
import datetime
import time
from pathlib import Path
from typing import List, Dict, Optional, Any

import chromadb
import ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

COLLECTION_NAME = "DOC_CHAT"
VDB_PATH = "./chroma_db"
CHAT_MODEL = "gpt-oss:20b-cloud"
DISTANCE_THRESHOLD = 1.2

EMBED_MODEL = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def chroma_init() -> chromadb.Collection:
    """
    Initializes the persistent ChromaDB client and retrieves or creates the target collection.
    
    Returns:
        chromadb.Collection: The initialized ChromaDB collection object.
    """
    client = chromadb.PersistentClient(path=VDB_PATH)
    return client.get_or_create_collection(name=COLLECTION_NAME)

def retrieve_with_sources(query: str, collection: chromadb.Collection) -> str:
    """
    Performs a semantic search against the collection and filters results based on a distance threshold.
    
    Args:
        query: The user's natural language question.
        collection: The ChromaDB collection to query.
        
    Returns:
        str: A formatted string of context parts with source attribution, 
             or an empty string if no relevant documents pass the threshold.
    """
    results = collection.query(
        query_texts=[query],
        n_results=3
    )
    
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]
    
    context_parts = []
    for doc, meta, dist in zip(docs, metas, distances):
        if dist <= DISTANCE_THRESHOLD:
            source_info = f"[Source: {meta.get('source')}]"
            context_parts.append(f"{source_info} {doc}")
    
    return "\n".join(context_parts)

def extract_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Reads a file and extracts its content along with filesystem metadata.
    
    Args:
        file_path: Absolute or relative path to the file.
        
    Returns:
        List[Dict]: A list containing a dictionary with 'content' and 'metadata' keys.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is not supported (.txt, .md).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
        
    ext = path.suffix.lower()
    if ext not in [".txt", ".md"]:
        raise ValueError(f"Unsupported file extension: {ext}")
        
    file_stat = path.stat()
    mod_date = datetime.datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
    
    base_metadata = {
        "source": path.name,
        "extension": ext,
        "last_modified": mod_date
    }
    
    with open(file_path, "r", encoding="utf-8") as f:
        return [{"content": f.read(), "metadata": base_metadata}]

def semantic_chunking(text: str) -> List[Any]:
    """
    Processes raw text through a two-stage pipeline: semantic splitting followed by recursive character refinement.
    
    Args:
        text: The raw string content to be chunked.
        
    Returns:
        List[Document]: A list of refined text chunks.
    """
    semantic_splitter = SemanticChunker(
        embeddings=EMBED_MODEL,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
    )
    
    final_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", "\t", " ", ""]
    )

    semantic_docs = semantic_splitter.create_documents([text])
    return final_splitter.split_documents(semantic_docs)

def file_dump(dir_path: str, collection: chromadb.Collection) -> None:
    """
    Scans a directory for files and ingests them into the vector database if they do not already exist.
    
    Args:
        dir_path: Path to the directory containing documents.
        collection: The ChromaDB collection where data will be stored.
    """
    path_obj = Path(dir_path)
    if not path_obj.exists():
        print(f"Error: Directory {dir_path} does not exist.")
        return

    file_paths = [str(f) for f in path_obj.iterdir() if f.is_file()]
    
    for f_path in file_paths:
        file_name = os.path.basename(f_path)
        
        existing = collection.get(where={"source": file_name}, limit=1)
        if existing["ids"]:
            print(f"Skipping {file_name}: Already exists.")
            continue
            
        try:
            raw_data = extract_data(f_path)
            print(f"Processing {file_name}...")
            
            chunks = semantic_chunking(raw_data[0]['content'])
            
            ids, docs, metas = [], [], []
            for i, chunk in enumerate(chunks):
                ids.append(f"{file_name}_chunk_{i+1}")
                docs.append(chunk.page_content)
                metas.append({**raw_data[0]['metadata'], "chunk_index": i+1})
            
            if ids:
                collection.upsert(ids=ids, documents=docs, metadatas=metas)
                print(f"Successfully added {file_name} ({len(chunks)} chunks)")
                
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

def chat_window(collection: chromadb.Collection) -> None:
    """
    Handles the interactive CLI loop for querying the RAG system.
    
    Args:
        collection: The initialized ChromaDB collection for retrieval.
    """
    print("\nAgent ready! Type 'exit' or 'bye' to quit.")
    
    while True:
        query = input("\n-> ").strip()
        if query.lower() in ["bye", "exit"]:
            print("AI-> Seeya!")
            break
            
        if not query:
            continue
            
        context = retrieve_with_sources(query, collection)
        
        system_prompt = (
            "You are a strict Document Assistant. Answer ONLY using the provided context. "
            "If the context is empty or insufficient, say: 'I don't have enough information in the documents.' "
            "Cite the specific document name within ' '. Reply only in simple text, no markdown."
        )
        user_prompt = f"CONTEXT:\n{context}\n\nQUESTION: {query}"
        
        response = ollama.chat(model=CHAT_MODEL, messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        print(f"\nAI-> {response['message']['content']}")

def main(file_dir: Optional[str]) -> None:
    """
    Main entry point that coordinates ingestion and starts the chat interface.
    
    Args:
        file_dir: Optional directory path provided via CLI arguments.
    """
    collection = chroma_init()
    
    if file_dir:
        start_time = time.perf_counter()
        file_dump(file_dir, collection)
        print(f"Ingestion complete in {time.perf_counter() - start_time:.4f} seconds.")
        
    chat_window(collection)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The Semantic Memory Engine")
    parser.add_argument("-fd", "--file_dir", type=str, help="Directory to ingest")
    args = parser.parse_args()
    
    try:
        main(args.file_dir)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Critical Error: {e}")
