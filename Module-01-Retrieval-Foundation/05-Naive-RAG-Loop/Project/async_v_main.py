import asyncio, argparse, os, datetime
from pathlib import Path
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
import ollama
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time

collection_name = "DOC_CHAT"
VDB_loc = f"./chroma_db"
CHAT_MODEL = "gpt-oss:20b-cloud"

# Initialize Embedding model once globally to save VRAM/Time
EMBED_MODEL = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

###-----CHROMA-DB-----###
def chroma_init():
    client = chromadb.PersistentClient(path=VDB_loc)
    return client.get_or_create_collection(name=collection_name)

def retrieve_with_sources(query, collection):
    results = collection.query(query_texts=[query], n_results=2)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    context_parts = []
    for doc, meta in zip(docs, metas):
        source_info = f"[Source: {meta.get('source')}]"
        context_parts.append(f"{source_info} {doc}")
    return "\n".join(context_parts)

###-----FILE and DATA ENTRY-----###
def semantic_chunking(TEXT, percentile_threshold=95, chunk_size=500, chunk_overlap=50):
    semantic_splitter = SemanticChunker(
        embeddings=EMBED_MODEL,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=percentile_threshold,
    )
    final_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=lambda x: len(x.split()),
        separators=["\n\n", "\n", "\t", " ", ""]
    )
    docs = semantic_splitter.create_documents([TEXT])
    return final_splitter.split_documents(docs)

async def dump_to_vector_DB(collection, documents, metadatas):
    IDS, DOC, META = [], [], []
    for i, doc in enumerate(documents):
        chunk_id = f"{metadatas['source']}_chunk_{i+1}"
        IDS.append(chunk_id)
        DOC.append(doc.page_content)
        META.append({**metadatas, "chunk_index": i+1})
    
    if IDS:
        # Chroma's upsert is synchronous, so we offload it
        await asyncio.to_thread(collection.upsert, ids=IDS, documents=DOC, metadatas=META)

async def process_file(f_path, collection, semaphore):
    async with semaphore: # Limits how many files are chunked at once
        file_name = os.path.basename(f_path)
        
        # Check if exists
        existing = await asyncio.to_thread(collection.get, where={"source": file_name}, limit=1)
        if existing["ids"]:
            print(f"Skipping {file_name}: Already exists.")
            return

        try:
            data = extract_data(f_path)[0] # Get first doc
            print(f"--> Chunking: {file_name}")
            
            # Offload heavy CPU work to a thread so loop doesn't block
            chunks = await asyncio.to_thread(semantic_chunking, data['content'])
            
            await dump_to_vector_DB(collection, chunks, data['metadata'])
            print(f"Successfully added {file_name} ({len(chunks)} chunks)")
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

async def file_dump(dir_path, collection):
    path_obj = Path(dir_path)
    if not path_obj.exists():
        print(f"Error: Directory {dir_path} not found.")
        return

    file_paths = [str(f) for f in path_obj.iterdir() if f.is_file()]
    print(f"Found {len(file_paths)} files. Starting ingestion...")
    
    semaphore = asyncio.Semaphore(3) # Process 3 files at a time to save CPU/RAM
    tasks = [process_file(fp, collection, semaphore) for fp in file_paths]
    await asyncio.gather(*tasks)

def extract_data(file):
    path = Path(file)
    ext = path.suffix.lower()
    file_stat = path.stat()
    mod_date = datetime.datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
    base_metadata = {"source": path.name, "extension": ext, "last_modified": mod_date}
    
    if ext in [".txt", ".md"]:
        with open(file, "r", encoding="utf-8") as f:
            return [{"content": f.read(), "metadata": base_metadata}]
    raise ValueError(f"Unsupported extension: {ext}")

###-----LLM BLOCK-----###
def chat_window(VDB):
    print("\nAI-> Hi, I'm NV7. Ready for your questions!")
    while True:
        query = input("\n-> ")
        if query.lower() in ["bye", "exit"]:
            print("AI-> Seeya!")
            break
        
        context = retrieve_with_sources(query, VDB)
        system_prompt = (
            "You are a strict Document Assistant. Answer ONLY using the provided context. "
            "If info is missing, say: 'I don't have enough information in the documents.' "
            "Cite sources inside ' '. Reply in simple text, no markdown."
        )
        user_prompt = f"CONTEXT:\n{context}\n\nQUESTION: {query}"
        
        response = ollama.chat(model=CHAT_MODEL, messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        print(f"AI-> {response['message']['content']}")

###-----MAIN CONTROLLER-----###
async def main(file_dir):
    collection = chroma_init()
    if file_dir:
        start_time = time.perf_counter()
        await file_dump(file_dir, collection)
        end_time = time.perf_counter()
        duration = end_time - start_time
        print(f"Data ingested in {duration:.4f} seconds")
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
