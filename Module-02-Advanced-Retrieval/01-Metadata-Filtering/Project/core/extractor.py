import os, datetime, uuid
from pathlib import Path
from typing import List,Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter

def read_data(file_path:str)->str:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
        
    ext = path.suffix.lower()
    if ext not in [".txt", ".md"]:
        raise ValueError(f"Unsupported file extension: {ext}")
        
    file_stat = path.stat()
    mod_date = datetime.datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
    
    base_metadata = {
        "source": path.name,
        "extension": ext,
        "last_modified": mod_date
    }
    
    with open(file_path, "r", encoding="utf-8") as f:
        return {"content": f.read(), "metadata": base_metadata}

class Chunker:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4o",
            separators = ["\n\n", "\n", " ", ""],
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap
        )

    def chunk(self, raw_text: str) -> Tuple[List[str], List[str]]:
        chunks = self.splitter.split_text(raw_text)
        # Generate a unique prefix for this batch to avoid ID collisions
        batch_id = uuid.uuid4().hex[:6]
        chunk_ids = [f"chk_{batch_id}_{i}" for i in range(len(chunks))]
        
        return chunks, chunk_ids