import chromadb
from typing import List,Optional

class Chroma_VDB():
    def __init__(
            self,DB_PATH:str,
            collection_name:str = "Smart_Document_CHAT"
            ):
        self.client = chromadb.PersistentClient(path=DB_PATH)
        self.collection = self.client.get_or_create_collection(name=collection_name)
    
    def file_check_query(self,file_name:str)-> bool:
        existing = self.collection.get(where={"source": file_name}, limit=1)
        if existing["ids"]:
            print(f"Skipping {file_name}, already exists in DB!")
            return True
        else:
            return False
    
    def context_query(self, query:str, filter:Optional[dict] = None, n_results: int = 3)-> chromadb.QueryResult:
        """
        Search the collection for relevant documents, optionally applying a metadata filter.
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=filter
            )
            return results
        except Exception as e:
            print(f"Error during context query: {e}")
            return {"ids": [], "documents": [[]], "metadatas": [[]]}

    def get_unique_values(self, key: str) -> List[str]:
        """
        Retrieves all unique values for a specific metadata key in the collection.
        Useful for providing 'context' to the LLM for filter generation.
        """
        try:
            results = self.collection.get(include=['metadatas'])
            metadatas = results['metadatas']
            unique_values = {m.get(key) for m in metadatas if m and m.get(key)}
            return sorted(list(unique_values))
        except Exception as e:
            print(f"Error fetching unique values for {key}: {e}")
            return []

    def context_ingest(self, ids:List[str], docs:List[str], meta:List[dict])-> None:
        """
        Add documents and their associated metadata to the ChromaDB collection.
        """
        try:
            self.collection.add(
                ids=ids,
                documents=docs,
                metadatas=meta
            )
        except Exception as e:
            print(f"Error during context ingest: {e}")
            raise e