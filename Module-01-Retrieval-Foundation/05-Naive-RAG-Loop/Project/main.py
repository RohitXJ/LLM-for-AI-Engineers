import asyncio, argparse, os, datetime
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

collection_name = "DOC_CHAT"
VDB_loc = f"./chroma_db"

###-----CHROMA-DB-----###

def chroma_init()->object:
    """
    Initializes the chromadb
    """
    client = chromadb.PersistentClient(path=VDB_loc)
    default_ef = embedding_functions.DefaultEmbeddingFunction()  
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=default_ef
    )
    global EMBED_MODEL
    EMBED_MODEL = collection._embedding_function
    return collection

###-----FILE and DATA ENTRY-----###

def file_dump(dir_path, collection):
    dir_path = Path(dir_path)
    file_paths = [str(f) for f in dir_path.iterdir() if f.is_file()]
    
    #Put to Async later
    def file_injested_chk(filename):
        existing = collection.get(where={"source": filename}, limit=1)
        if existing["ids"]:
            print(f"Skipping {filename}: Already exists in DB.")
            return True
        else:
            print(f"Dumping {filename} into DB.")
            return False
    #Put to Async later
    def dump_to_vector_DB(documents,metadatas):
        print(documents)

    #Put to Async later
    print(file_paths)
    for f in file_paths:
        if not file_injested_chk(os.path.basename(f)):
            documents = injest_data_pool(f)
            chunks = semantic_chunking(TEXT=documents['content'])
            dump_to_vector_DB(documents=chunks,metadatas=documents['metadata'])


def semantic_chunking(TEXT,percentile_threshold=95,chunk_size=500,chunk_overlap=50):
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_text_splitters import RecursiveCharacterTextSplitter
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
        separators=["\n\n", "\n", " ", ""]
    )

    # Execution
    docs = semantic_splitter.create_documents([TEXT])
    final_chunks = final_splitter.split_documents(docs)
    return final_chunks


def injest_data_pool(file):
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

###-----MAIN CONTROLLER-----###

def main(file_dir):
    collection = chroma_init()
    if file_dir:
        file_dump(file_dir,collection)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The Semantic Memory Engine")
    parser.add_argument("-fd", "--file_dir", type=str, help="Enter your file dir to injest the required files")
    args = parser.parse_args()
    
    try:
        main(args.file_dir)
        #asyncio.run(main(args.file_dir))
    except Exception as e:
        print(f"Critical Error: {e}")