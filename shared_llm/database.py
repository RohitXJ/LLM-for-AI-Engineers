import chromadb
import uuid
from typing import List, Optional, Dict, Any, Set

class ChromaManager:
    """
    Orchestrates interaction with ChromaDB for document storage and retrieval.
    Supports persistent storage, metadata filtering, and collection management.
    """
    def __init__(self, path: str, collection_name: str = "default_collection"):
        """
        Initializes the ChromaManager and connects to the persistent client.
        
        Args:
            path (str): Local directory path where ChromaDB data is stored.
            collection_name (str): Name of the collection to work with.
        """
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def file_exists(self, source_name: str) -> bool:
        """
        Checks if a file (by its source metadata) already exists in the collection.
        
        Args:
            source_name (str): The filename/source to check for.
            
        Returns:
            bool: True if at least one chunk from this source exists.
        """
        existing = self.collection.get(where={"source": source_name}, limit=1)
        return len(existing["ids"]) > 0

    def get_all_ids(self) -> Set[str]:
        """
        Fetches every document ID currently in the collection.
        
        Returns:
            Set[str]: A set of all document IDs.
        """
        results = self.collection.get(include=[])
        return set(results['ids'])

    def ingest(self, documents: List[str], metadatas: List[Dict[str, Any]], ids: Optional[List[str]] = None) -> None:
        """
        Adds a batch of documents and metadata to the collection.
        
        Args:
            documents (List[str]): The text content of each chunk.
            metadatas (List[Dict[str, Any]]): Metadata dictionaries for each chunk.
            ids (Optional[List[str]]): Unique IDs for each chunk. Generated if missing.
        """
        if ids is None:
            ids = [f"chk_{uuid.uuid4().hex[:10]}" for _ in documents]
            
        try:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
        except Exception as e:
            raise RuntimeError(f"ChromaDB Ingest Failed: {e}")

    def add_documents(self, documents: List[Dict[str, Any]], chunker: Optional[Any] = None) -> None:
        """
        High-level method to process and add documents to the collection.
        If a chunker is provided, it handles splitting and mapping metadata.
        
        Args:
            documents (List[Dict[str, Any]]): List of documents (output of DataLoader).
            chunker (Optional[Chunker]): Optional Chunker instance for text splitting.
        """
        all_ids = []
        all_texts = []
        all_metadatas = []

        for doc in documents:
            content = doc["content"]
            metadata = doc["metadata"]
            
            if chunker:
                chunks, ids = chunker.split(content)
                for c_id, chunk in zip(ids, chunks):
                    all_ids.append(c_id)
                    all_texts.append(chunk)
                    # Merge doc metadata into chunk metadata
                    all_metadatas.append(metadata.copy())
            else:
                doc_id = metadata.get("id") or f"doc_{uuid.uuid4().hex[:10]}"
                all_ids.append(doc_id)
                all_texts.append(content)
                all_metadatas.append(metadata)
        
        if all_texts:
            self.ingest(documents=all_texts, metadatas=all_metadatas, ids=all_ids)

    def query(self, query_text: str, n_results: int = 5, where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Performs a semantic search in the collection, optionally filtered by metadata.
        
        Args:
            query_text (str): The search query.
            n_results (int): Number of top matches to return.
            where (Optional[Dict[str, Any]]): ChromaDB-compatible metadata filter dictionary.
            
        Returns:
            Dict[str, Any]: The raw ChromaDB query result payload.
        """
        return self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where
        )

    def get_unique_metadata_values(self, key: str) -> List[Any]:
        """
        Scans the collection to find all unique values for a specific metadata key.
        Useful for guiding LLMs during filter generation (tag awareness).
        
        Args:
            key (str): The metadata key to inspect (e.g., 'topic', 'year').
            
        Returns:
            List[Any]: Sorted list of unique values found.
        """
        try:
            results = self.collection.get(include=['metadatas'])
            metadatas = results['metadatas']
            unique_values = {m.get(key) for m in metadatas if m and m.get(key) is not None}
            return sorted(list(unique_values))
        except Exception as e:
            print(f"Warning: Error fetching unique values for '{key}': {e}")
            return []

    def get_filter_options(self, keys: List[str]) -> Dict[str, List[Any]]:
        """
        Convenience method to get unique values for multiple metadata keys at once.
        
        Args:
            keys (List[str]): List of metadata keys to inspect.
            
        Returns:
            Dict[str, List[Any]]: Map of key -> unique_values.
        """
        return {key: self.get_unique_metadata_values(key) for key in keys}
