from pathlib import Path
from core import *
import os, uuid

COLLECTION_NAME = "SMART_DOC_CHAT"
VDB_PATH = "./chroma_db"
LLM = Ollama(model="gpt-oss:20b-cloud")
chunker = Chunker()
chroma = Chroma_VDB(DB_PATH=VDB_PATH, collection_name=COLLECTION_NAME)

def main(path):
    dir_path = Path(path)
    if not dir_path.exists():
        raise FileNotFoundError(f"File not found: {dir_path}")
    
    file_names = [str(f) for f in dir_path.iterdir() if f.is_file()]

    for file_name in file_names:
        raw_data = read_data(file_path=file_name)

        print(f"\n--- Processing: {os.path.basename(file_name)} ---")

        if not chroma.file_check_query(os.path.basename(file_name)):
            # Step 1: Extract Global Anchor Metadata
            global_meta = LLM.extract_document_metadata(content=raw_data['content'])
            
            # Step 2: Chunk and Ingest with Global Anchor
            chunks, ids = chunker.chunk(raw_text=raw_data['content'])
            data_ingest(chunks=chunks, ch_ids=ids, file_meta={**global_meta, **raw_data["metadata"]})

    print("\n--- OmniSearch Ready! (Type 'exit' to quit) ---")
    while True:
        query = input("<- Query: ")
        if query.lower() in ["bye", "exit"]:
            break
        
        # Step 3: Fetch existing tags to guide the LLM
        existing_topics = chroma.get_unique_values("topic")
        existing_years = chroma.get_unique_values("year")
        
        # Step 4: Self-Querying Retrieval with Tag Awareness
        filters = LLM.generate_filter(
            user_query=query, 
            existing_metadata={"topic": existing_topics, "year": existing_years}
        )
        print(f"[Logic] Applied Filters: {filters}")
        
        results = chroma.context_query(query=query, filter=filters)
        
        if not results['documents'][0]:
            print("[AI]: No relevant information found in the database with those filters.")
            continue
            
        context = "\n\n".join(results['documents'][0])
        LLM.answer_question(context=context, query=query)

def data_ingest(chunks: list[str], ch_ids: list[str], file_meta: dict)->None:
    ids = []
    metadata = []
    for chunk, c_id in zip(chunks, ch_ids):
        ids.append(f"{file_meta['source']}_{c_id}")
        # Extract local nuance while respecting the global anchor
        meta_ext = LLM.extract_metadata(context=chunk, global_meta=file_meta)
        metadata.append({**meta_ext, **file_meta})
    
    try:
        chroma.context_ingest(
            ids=ids,
            docs=chunks,
            meta=metadata
            )
        print(f"Successfully ingested {file_meta['source']}")
    except Exception as e:
        print(f"Can't ingest data for {file_meta['source']} because : {e}")

if __name__ == "__main__":
    main(path = f"data")