import chromadb

class Chroma:
    def __init__(self, loc, c_name="doc_chat"):
        self.client = chromadb.PersistentClient(path=loc)
        self.collection = self.client.get_or_create_collection(c_name)

    def get_filter_values(self) -> dict:
        """
        Fetches all metadata from the collection and extracts unique 
        values for 'department' and 'category'.
        """
        results = self.collection.get(include=["metadatas"])
        metadatas = results.get("metadatas", [])
        
        # Extract unique values while ignoring None
        departments = {m.get("department") for m in metadatas if m.get("department")}
        categories = {m.get("category") for m in metadatas if m.get("category")}
        
        return {
            "department": sorted(list(departments)),
            "category": sorted(list(categories))
        }
    def query_context(self,query_text, filter_values: dict = {}) -> str:
        """
        Queries the collection with a given query text and filters based on provided metadata.
        """
        results = self.collection.query(
            query_texts = query_text,
            where=filter_values,
            n_results=50
        )
        return results
    
    def data_ingest(self,new_ids: list, corpus: list[dict]) -> None:
        """
        Inserts a batch of documents into the collection with their corresponding metadata.
        """
        ids = []
        metadata = []
        doc = []

        for item in corpus:
            # Check if the item's ID is in the target list
            if item.get("id") in new_ids:
                doc.append(
                    f"TITLE: {item.get('title', 'N/A')} CONTENT: {item.get('content', '')}"
                )
                metadata.append(
                    {
                        "department": item.get("department"),
                        "category": item.get("category"),
                        "year": item.get("year"),
                    }
                )
                ids.append(item.get("id"))
        
        self.collection.add(
            ids=ids,
            documents=doc,
            metadatas=metadata
        )
        print("Data Ingested Successfully")
    
    def id_fetch(self):
        """
        Fetches all document IDs currently in the collection.
        
        Returns:
            set: A set of existing document IDs.
        """
        results = self.collection.get(include=[])
        return set(results['ids'])


