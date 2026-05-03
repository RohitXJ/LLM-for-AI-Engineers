from pathlib import Path
from core import *
def main(path):
    #Placeholder for LLM, VectorDB objects
    dir_path = Path(path)
    file_names = [str(f) for f in dir_path.iterdir() if f.is_file()]
    chunker = Chunker()
    
    for file_name in file_names:
        raw_data = read_data(file_path=file_name)
        print(f"RAW DATA\n{raw_data}\n")
        chunks,ids = chunker.chunk(raw_text=raw_data['content'])
        for ch,i in zip(chunks,ids):
            print(f"Chunk Data\n{ch}\nChunk ID\n{i}\n")

    
    

if __name__ == "__main__":
    main(path = f"data")