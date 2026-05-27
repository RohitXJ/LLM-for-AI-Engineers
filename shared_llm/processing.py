import os
import datetime
import uuid
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

class Chunker:
    """
    Handles text splitting into manageable chunks for vector database ingestion.
    """
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50, model_name: str = "gpt-4o"):
        """
        Initializes the Chunker with specific splitting parameters.
        
        Args:
            chunk_size (int): Max number of tokens/characters per chunk.
            chunk_overlap (int): Overlap between chunks to maintain context.
            model_name (str): The tiktoken encoder model to use for counting.
        """
        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name=model_name,
            separators=["\n\n", "\n", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def split(self, text: str) -> Tuple[List[str], List[str]]:
        """
        Splits raw text and generates unique IDs for each chunk.
        
        Args:
            text (str): The raw text to split.
            
        Returns:
            Tuple[List[str], List[str]]: A list of text chunks and a list of unique IDs.
        """
        chunks = self.splitter.split_text(text)
        batch_id = uuid.uuid4().hex[:6]
        chunk_ids = [f"chk_{batch_id}_{i}" for i in range(len(chunks))]
        return chunks, chunk_ids

class DataLoader:
    """
    Utilities for reading and aggregating data from various file formats.
    """
    @staticmethod
    def read_file(file_path: str) -> Dict[str, Any]:
        """
        Reads a single text or markdown file and returns its content and basic metadata.
        
        Args:
            file_path (str): Path to the file.
            
        Returns:
            Dict[str, Any]: Contains 'content' and 'metadata' (source, extension, last_modified).
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        ext = path.suffix.lower()
        if ext not in [".txt", ".md"]:
            raise ValueError(f"Unsupported file extension: {ext}")
            
        file_stat = path.stat()
        mod_date = datetime.datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        
        metadata = {
            "source": path.name,
            "extension": ext,
            "last_modified": mod_date
        }
        
        with open(file_path, "r", encoding="utf-8") as f:
            return {"content": f.read(), "metadata": metadata}

    @staticmethod
    def load_json_directory(directory_path: str) -> List[Dict[str, Any]]:
        """
        Aggregates all JSON files from a directory into a single list of dictionaries.
        
        Args:
            directory_path (str): Path to the directory containing JSON files.
            
        Returns:
            List[Dict[str, Any]]: List of all documents found.
        """
        all_documents = []
        data_path = Path(directory_path)
        
        if not data_path.exists():
            return []

        for file_path in data_path.glob("*.json"):
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_documents.extend(data)
                    elif isinstance(data, dict):
                        all_documents.append(data)
                except json.JSONDecodeError:
                    print(f"Warning: {file_path.name} is not a valid JSON. Skipping.")
        
        return all_documents
