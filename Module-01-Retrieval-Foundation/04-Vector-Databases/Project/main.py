import argparse
import asyncio
import os
from helper import knowledge_chk, extract_text, dump_to_vector_DB, chroma_init

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_PATH = os.path.join(SCRIPT_DIR, "knowledge_base")

async def main(query):
    print("--- Initializing In-Memory Vector DB (Auto-Embedding Enabled) ---")
    collection = chroma_init()
    
    print(f"--- Scanning knowledge base at {KNOWLEDGE_PATH} ---")
    file_list = knowledge_chk(path=KNOWLEDGE_PATH)
    
    if not file_list:
        print("No files found in knowledge base.")
        return

    async def data_pool_entry(file_path):
        print(f"Processing: {os.path.basename(file_path)}")
        try:
            # extract_text returns a list of page/document objects
            documents = await extract_text(file_path)
            # dump_to_vector_DB handles chunking and uploading using collection's embedding_fn
            await dump_to_vector_DB(collection, documents)
            return True
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return False

    print(f"--- Ingesting Documents ---")
    results = await asyncio.gather(*(data_pool_entry(f) for f in file_list))
    
    print(f"--- Searching for: '{query}' ---")
    # Query using query_texts; Chroma will handle the embedding automatically
    results = collection.query(
        query_texts=[query],
        n_results=3
    )
    
    print("\n--- Top Search Results ---")
    if results['ids'] and results['ids'][0]:
        for i in range(len(results['ids'][0])):
            print(f"\nResult {i+1} (ID: {results['ids'][0][i]}):")
            print(f"Score: {results['distances'][0][i]}")
            print(f"Source: {results['metadatas'][0][i].get('source')} (Page: {results['metadatas'][0][i].get('page')})")
            print(f"Content: {results['documents'][0][i][:200]}...")
    else:
        print("No matches found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The Semantic Memory Engine")
    parser.add_argument("-q", "--query", type=str, help="Enter your query to be searched")

    args = parser.parse_args()
    
    query_text = args.query if args.query else "What is the main topic of the documents?"
    
    try:
        asyncio.run(main(query=query_text))
    except Exception as e:
        print(f"Critical Error: {e}")
