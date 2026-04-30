import argparse, os, datetime,time
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
import ollama
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

collection_name = "DOC_CHAT"
VDB_loc = f"./chroma_db"
EMBED_MODEL = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
CHAT_MODEL = "gpt-oss:20b-cloud"

###-----CHROMA-DB-----###

def chroma_init()->object:
    """
    Initializes the chromadb
    """
    client = chromadb.PersistentClient(path=VDB_loc)
    collection = client.get_or_create_collection(
        name=collection_name
    )
    return collection

def retrieve_with_sources(query, collection):
    """Querying is simpler: just pass the string text."""
    results = collection.query(
        query_texts=[query],
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

###-----FILE and DATA ENTRY-----###

def file_dump(dir_path, collection):
    dir_path = Path(dir_path)
    if not dir_path.exists():
        print(f"Error: Directory {dir_path} does not exist.")
        return

    file_paths = [str(f) for f in dir_path.iterdir() if f.is_file()]
    
    #Put to Async later
    def file_ingested_chk(filename):
        existing = collection.get(where={"source": filename}, limit=1)
        if existing["ids"]:
            print(f"Skipping {filename}: Already exists in DB.")
            return True
        else:
            return False
    #Put to Async later
    def dump_to_vector_DB(documents,metadatas):
        IDS = []
        DOC = []
        META = []
        for i,doc in enumerate(documents):
            chunk_id = f"{metadatas['source']}_chunk_{i+1}"
            IDS.append(chunk_id)
            DOC.append(doc.page_content)
            META.append({**metadatas,"chunk_index":i+1})
        
        if IDS:
            collection.upsert(
            ids=IDS,
            documents=DOC,
            metadatas=META
            )

    #Put to Async later
    print(f"Found {len(file_paths)} files in {dir_path}")
    for f in file_paths:
        file_name = os.path.basename(f)
        if not file_ingested_chk(file_name):
            try:
                documents = extract_data(f)
                print(f"Dumping {file_name} into DB...")
                chunks = semantic_chunking(TEXT=documents[0]['content'])
                dump_to_vector_DB(documents=chunks,metadatas=documents[0]['metadata'])
                print(f"{file_name} added to DB successfully ({len(chunks)} chunks)")
            except ValueError as ve:
                print(f"Skipping {file_name}: {ve}")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")


def semantic_chunking(TEXT,percentile_threshold=95,chunk_size=500,chunk_overlap=50):
    # 1. Semantic Splitting (Meaning-based)
    semantic_splitter = SemanticChunker(
        embeddings=EMBED_MODEL, 
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=percentile_threshold,
    )
    # 2. Refined Splitting (Size & Overlap-based)
    final_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=lambda x: len(x.split()), # Force word-count instead of characters
        separators=["\n\n", "\n", "\t", " ", ""] 
    )

    # Execution
    docs = semantic_splitter.create_documents([TEXT])
    final_chunks = final_splitter.split_documents(docs)
    return final_chunks


def extract_data(file):
    path = Path(file)
    ext = path.suffix.lower()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file}")
    # File-level metadata
    file_stat = path.stat()
    mod_date = datetime.datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
    
    base_metadata = {
        "source": path.name,
        "extension": ext,
        "last_modified": mod_date
    }
    documents = []

    try:
        if ext in [".txt", ".md"]:
            with open(file, "r", encoding="utf-8") as f:
                content = f.read()
                documents.append({
                    "content": content,
                    "metadata": {**base_metadata}
                })
        else:
            # Silently ignore unsupported files when batching, or raise for single file
            raise ValueError(f"Unsupported file extension: {ext}")

        return documents

    except Exception as e:
        raise Exception(f"Error processing {file}: {str(e)}")

###-----LLM BLOCK-----###

def chat_window(VDB):
    query = ''
    while True:
        query = input("\n-> ")
        if query.lower() == ("bye" or "exit"):
            print("\nAI-> Seeya!")
            break
        context = retrieve_with_sources(query,VDB)
        system_prompt = (
            "You are a strict Document Assistant. Answer ONLY using the provided context. "
            "If the answer is missing from the context, say: 'I don't have enough information in the documents.' "
            "Always cite your source by mentioning the specific section or document name within ' '. "
            "Do not use external knowledge. Be concise and do not answer with unnecessary context info."
            "You can be asked to summarize,so do so when asked for."
            "Reply only in simple text, no markdown."
        )
        user_prompt = f"CONTEXT:\n{context}\n\nQUESTION: {query}"
        
        print("\n")
        
        response = ollama.chat(model=CHAT_MODEL, messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        print(f"AI-> {response['message']['content']}")

###-----MAIN CONTROLLER-----###

def main(file_dir):
    collection = chroma_init()
    if file_dir:
        start_time = time.perf_counter()
        file_dump(file_dir,collection)
        end_time = time.perf_counter()
        duration = end_time - start_time
        print(f"Data ingested in {duration:.4f} seconds")
    print("Agent is ready to chat with your files!\n")
    print("Hi,I'm NV7,\nready to assist you with your queries from your documents!")
    chat_window(collection)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The Semantic Memory Engine")
    parser.add_argument("-fd", "--file_dir", type=str, help="Enter your file dir to injest the required files")
    args = parser.parse_args()
    
    try:
        main(args.file_dir)
        #asyncio.run(main(args.file_dir))
    except Exception as e:
        print(f"Critical Error: {e}")