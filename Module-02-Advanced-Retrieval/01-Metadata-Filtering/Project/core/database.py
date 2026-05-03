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
    
    def context_query(self, query:str, filter:List[dict])-> chromadb.QueryResult:
        pass

    def context_ingest(self, ids:List[str], docs:List[str], meta:List[dict])-> None:
        pass