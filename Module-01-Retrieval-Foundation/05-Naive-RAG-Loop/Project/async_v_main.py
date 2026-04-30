import asyncio
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
MAX_CONCURRENT_FILES = 3

EMBED_MODEL = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def chroma_init() -> chromadb.Collection:
    """
    Initializes the persistent ChromaDB client.
    
    Returns:
        chromadb.Collection: The initialized collection.
    """
    client = chromadb.PersistentClient(path=VDB_PATH)
    return client.get_or_create_collection(name=COLLECTION_NAME)

def retrieve_with_sources(query: str, collection: chromadb.Collection) -> str:
    """
    Synchronous retrieval with distance thresholding.
    
    Args:
        query: User input string.
        collection: Vector database collection.
        
    Returns:
        str: Formatted context string.
    """
    results = collection.query(query_texts=[query], n_results=3)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]
    
    context_parts = []
    for doc, meta, dist in zip(docs, metas, distances):
        if dist <= DISTANCE_THRESHOLD:
            context_parts.append(f"[Source: {meta.get('source')}] {doc}")
            
    return "\n".join(context_parts)

def extract_data(file_path: str) -> Dict[str, Any]:
    """
    Extracts text and metadata from a file.
    
    Args:
        file_path: Path to file.
        
    Returns:
        Dict: Content and metadata.
    """
    path = Path(file_path)
    ext = path.suffix.lower()
    file_stat = path.stat()
    mod_date = datetime.datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    return {
        "content": content,
        "metadata": {
            "source": path.name,
            "extension": ext,
            "last_modified": mod_date
        }
    }

def semantic_chunking(text: str) -> List[Any]:
    """
    Performs hybrid semantic and character-based chunking.
    
    Args:
        text: Input text.
        
    Returns:
        List: Chunks of text.
    """
    semantic_splitter = SemanticChunker(embeddings=EMBED_MODEL)
    final_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    docs = semantic_splitter.create_documents([text])
    return final_splitter.split_documents(docs)

async def process_file(f_path: str, collection: chromadb.Collection, semaphore: asyncio.Semaphore) -> None:
    """
    Orchestrates the asynchronous processing of a single file including chunking and DB insertion.
    
    Args:
        f_path: Path to the target file.
        collection: The ChromaDB collection.
        semaphore: Concurrency control primitive.
    """
    async with semaphore:
        file_name = os.path.basename(f_path)
        
        existing = await asyncio.to_thread(collection.get, where={"source": file_name}, limit=1)
        if existing["ids"]:
            print(f"Skipping {file_name}: Already exists.")
            return

        try:
            data = await asyncio.to_thread(extract_data, f_path)
            print(f"--> Chunking: {file_name}")
            
            chunks = await asyncio.to_thread(semantic_chunking, data['content'])
            
            ids, docs, metas = [], [], []
            for i, chunk in enumerate(chunks):
                ids.append(f"{file_name}_chunk_{i+1}")
                docs.append(chunk.page_content)
                metas.append({**data['metadata'], "chunk_index": i+1})
            
            if ids:
                await asyncio.to_thread(collection.upsert, ids=ids, documents=docs, metadatas=metas)
                print(f"Done: {file_name} ({len(chunks)} chunks)")
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

async def file_dump(dir_path: str, collection: chromadb.Collection) -> None:
    """
    Manages the batch ingestion of files using an asynchronous task pool.
    
    Args:
        dir_path: Source directory.
        collection: Target collection.
    """
    path_obj = Path(dir_path)
    if not path_obj.exists():
        print(f"Error: Directory {dir_path} not found.")
        return

    file_paths = [str(f) for f in path_obj.iterdir() if f.is_file() and f.suffix in [".txt", ".md"]]
    print(f"Found {len(file_paths)} files. Starting ingestion...")
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_FILES)
    tasks = [process_file(fp, collection, semaphore) for fp in file_paths]
    await asyncio.gather(*tasks)

def chat_window(collection: chromadb.Collection) -> None:
    """
    Interactive chat interface.
    
    Args:
        collection: Vector DB collection.
    """
    print("\nNV7 Ready. Type 'exit' to quit.")
    while True:
        query = input("\n-> ").strip()
        if query.lower() in ["bye", "exit"]:
            break
        
        context = retrieve_with_sources(query, collection)
        
        system_prompt = (
            "You are a strict Document Assistant. Answer ONLY using the provided context. "
            "If info is missing, say: 'I don't have enough information in the documents.' "
            "Cite sources inside ' '. Simple text only."
        )
        
        response = ollama.chat(model=CHAT_MODEL, messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {query}"}
        ])
        print(f"AI-> {response['message']['content']}")

async def main(file_dir: Optional[str]) -> None:
    """
    Async application controller.
    
    Args:
        file_dir: Directory path.
    """
    collection = chroma_init()
    if file_dir:
        start_time = time.perf_counter()
        await file_dump(file_dir, collection)
        print(f"Data ingested in {time.perf_counter() - start_time:.4f} seconds")
    chat_window(collection)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-fd", "--file_dir", type=str)
    args = parser.parse_args()
    try:
        asyncio.run(main(args.file_dir))
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Critical Error: {e}")
