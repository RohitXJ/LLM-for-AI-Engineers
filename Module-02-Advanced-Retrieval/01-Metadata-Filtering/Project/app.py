from pathlib import Path
from core import *
import os

COLLECTION_NAME = "SMART_DOC_CHAT"
VDB_PATH = "./chroma_db"

def main(path):
    #Placeholder for LLM objects
    chunker = Chunker()
    chroma = Chroma_VDB(DB_PATH=VDB_PATH, collection_name=COLLECTION_NAME)

    dir_path = Path(path)
    if not dir_path.exists():
        raise FileNotFoundError(f"File not found: {dir_path}")
    
    file_names = [str(f) for f in dir_path.iterdir() if f.is_file()]

    for file_name in file_names:
        raw_data = read_data(file_path=file_name)

        print(f"RAW DATA\n{raw_data}\n")#CHECKING

        if not chroma.file_check_query(os.path.basename(file_name)):
            chunks,ids = chunker.chunk(raw_text=raw_data['content'])

            for ch,i in zip(chunks,ids):#CHECKING
                print(f"Chunk Data\n{ch}\nChunk ID\n{i}\n")#CHECKING

    
    

if __name__ == "__main__":
    main(path = f"data")