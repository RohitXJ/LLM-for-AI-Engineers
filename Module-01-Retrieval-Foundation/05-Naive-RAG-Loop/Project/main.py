import asyncio, argparse, os, datetime
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
collection_name = "DOC_CHAT"
VDB_loc = "Module-01-Retrieval-Foundation\05-Naive-RAG-Loop\Project\chroma_db"

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
    return collection

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
            return False
    
    #Put to Async later
    print(file_paths)
    for f in file_paths:
        if not file_injested_chk(os.path.basename(f)):
            documents = injest_data_pool(f)
        print(documents)

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