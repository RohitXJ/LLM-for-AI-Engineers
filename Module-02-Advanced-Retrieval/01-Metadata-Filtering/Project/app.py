from pathlib import Path
from core import *
import os, uuid

COLLECTION_NAME = "SMART_DOC_CHAT"
VDB_PATH = "./chroma_db"
LLM = Gemini()
chunker = Chunker()
chroma = Chroma_VDB(DB_PATH=VDB_PATH, collection_name=COLLECTION_NAME)

def main(path):
    #Placeholder for LLM objects

    dir_path = Path(path)
    if not dir_path.exists():
        raise FileNotFoundError(f"File not found: {dir_path}")
    
    file_names = [str(f) for f in dir_path.iterdir() if f.is_file()]

    for file_name in file_names:
        raw_data = read_data(file_path=file_name)

        print(f"RAW DATA\n{raw_data}\n")#CHECKING

        if not chroma.file_check_query(os.path.basename(file_name)):
            chunks,ids = chunker.chunk(raw_text=raw_data['content'])
            data_ingest(chunks=chunks, ch_ids=ids, file_meta=raw_data["metadata"])

def data_ingest(chunks: list[str], ch_ids: list[str], file_meta: dict)->None:
    ids = []
    metadata = []
    for chunk,c_id in zip(chunks,ch_ids):
        ids.append(f"{file_meta['source']}_{c_id}")
        meta_ext = LLM.extract_metadata(context=chunk)
        metadata.append({**meta_ext,**file_meta})
    
    try:
        print(f"\nID:{ids}\ncontext:{chunks}\nmetadata:{metadata}")#CHECKING
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