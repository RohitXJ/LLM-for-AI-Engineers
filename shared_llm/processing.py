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
    def _get_file_metadata(path: Path) -> Dict[str, Any]:
        """Helper to extract standard file metadata."""
        file_stat = path.stat()
        mod_date = datetime.datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        return {
            "source": path.name,
            "extension": path.suffix.lower(),
            "last_modified": mod_date,
            "id": f"doc_{uuid.uuid4().hex[:8]}"
        }

    @staticmethod
    def read_file(file_path: str, call_in_lib: bool = False, path_obj: Path = None) -> Dict[str, Any]:
        """
        Reads a single text or markdown file and returns its content and basic metadata.
        
        Args:
            file_path (str): Path to the file.
            call_in_lib (bool): If True, uses the provided path_obj.
            path_obj (Path): Pre-constructed Path object.
            
        Returns:
            Dict[str, Any]: Contains 'content' and 'metadata' (source, extension, last_modified, id).
        """
        path = path_obj if (call_in_lib and path_obj) else Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
            
        ext = path.suffix.lower()
        if ext not in [".txt", ".md"]:
            raise ValueError(f"Unsupported file extension: {ext}")
            
        metadata = DataLoader._get_file_metadata(path)
        
        with open(path, "r", encoding="utf-8") as f:
            return {"content": f.read(), "metadata": metadata}

    @staticmethod
    def read_json(file_path: str, call_in_lib: bool = False, path_obj: Path = None) -> List[Dict[str, Any]]:
        """
        Reads a JSON file and returns its content in standardized document format.
        
        Args:
            file_path (str): Path to the file.
            call_in_lib (bool): If True, uses the provided path_obj.
            path_obj (Path): Pre-constructed Path object.
            
        Returns:
            List[Dict[str, Any]]: List of documents with 'content' and 'metadata'.
        """
        path = path_obj if (call_in_lib and path_obj) else Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
            
        file_metadata = DataLoader._get_file_metadata(path)
        documents = []

        with open(path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                items = data if isinstance(data, list) else [data]
                
                for i, item in enumerate(items):
                    if not isinstance(item, dict):
                        item = {"content": str(item)}
                    
                    # Ensure content key exists
                    if "content" not in item:
                        # Fallback: find any string field if content is missing, or use whole dict as string
                        potential_content = item.get("text") or item.get("body") or item.get("page_content")
                        item["content"] = potential_content if potential_content else str(item)
                    
                    # Merge metadata
                    existing_metadata = item.get("metadata", {})
                    if not isinstance(existing_metadata, dict):
                        existing_metadata = {"original_metadata": existing_metadata}
                    
                    # Combine file metadata with item metadata
                    merged_metadata = {**existing_metadata, **file_metadata}
                    
                    # Ensure unique ID for multiple items in one file
                    if len(items) > 1:
                        merged_metadata["id"] = f"{file_metadata['id']}_{i}"
                    
                    documents.append({
                        "content": item["content"],
                        "metadata": merged_metadata
                    })
                    
            except json.JSONDecodeError:
                print(f"Warning: {path.name} is not a valid JSON. Skipping.")
        
        return documents

    _HANDLERS = {
        ".txt": "read_file",
        ".md": "read_file",
        ".json": "read_json"
    }

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
            all_documents.extend(DataLoader.read_json(str(file_path), call_in_lib=True, path_obj=file_path))
        
        return all_documents
    
    @classmethod
    def master_loader(cls, directory_path: str, allowed_files: List[str] = None) -> List[Dict[str, Any]]:
        """
        Extracts all data from all types of files in a directory.
        
        Args:
            directory_path (str): Path to the directory.
            allowed_files (List[str]): List of suffixes (e.g. ['.txt', '.md']).
            
        Returns:
            List[Dict[str, Any]]: Aggregated list of all extracted documents.
        """
        path = Path(directory_path)
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
            
        all_documents = []
        
        # If allowed_files is not provided, use all supported extensions
        if allowed_files is None:
            allowed_files = list(cls._HANDLERS.keys())
        else:
            # Standardize suffixes to start with dot
            allowed_files = [ext if ext.startswith('.') else f'.{ext}' for ext in allowed_files]

        for file_path in path.iterdir():
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext in allowed_files and ext in cls._HANDLERS:
                    method_name = cls._HANDLERS[ext]
                    handler = getattr(cls, method_name)
                    try:
                        # Using call_in_lib and path_obj as requested
                        result = handler(str(file_path), call_in_lib=True, path_obj=file_path)
                        
                        if isinstance(result, list):
                            all_documents.extend(result)
                        else:
                            all_documents.append(result)
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
                        
        return all_documents
        