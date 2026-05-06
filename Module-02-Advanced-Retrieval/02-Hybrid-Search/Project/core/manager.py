import json
import pickle
from pathlib import Path
from .engines import KeywordEngine, VectorEngine

class IngestionManager:
    """
    Coordinates data ingestion, synchronization between JSON files and databases,
    and manages the persistence of search indices.
    """
    def __init__(self, chroma_loc, bm25_loc, data_loc):
        """
        Initializes the IngestionManager with paths and search engines.
        
        Args:
            chroma_loc (str): Path to ChromaDB storage.
            bm25_loc (str): Path to the BM25 pickle file.
            data_loc (str): Path to the directory containing JSON documents.
        """
        self.chroma_loc = chroma_loc
        self.bm25_loc = Path(bm25_loc)
        self.data_loc = data_loc
        
        # Ensure the directory for the BM25 index exists
        self.bm25_loc.parent.mkdir(parents=True, exist_ok=True)

        try:
            self.vec_eng = VectorEngine(loc=self.chroma_loc)
        except Exception as e:
            print(f"Error loading ChromaDB: {e}")
            self.vec_eng = None

        try:
            self.kw_eng = KeywordEngine()
        except Exception as e:
            print(f"Error initializing Keyword Engine: {e}")
            self.kw_eng = None

    def load_json_data(self):
        """
        Reads all JSON files from the data directory and aggregates them.
        
        Returns:
            list: A list of all documents found in the JSON files.
        """
        all_documents = []
        data_path = Path(self.data_loc)
        
        if not data_path.exists():
            print(f"Warning: Directory {self.data_loc} not found.")
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
                    print(f"Error: {file_path.name} is not a valid JSON file. Skipping.")
        
        return all_documents

    def sync(self):
        """
        Orchestrates the synchronization process.
        Checks for new documents, updates ChromaDB, and manages the BM25 index.
        """
        if not self.vec_eng or not self.kw_eng:
            print("Engines not properly initialized. Aborting sync.")
            return

        corpus = self.load_json_data()
        existing_ids = self.vec_eng.id_fetch()
        
        # Identify documents not yet in the Vector DB
        new_docs = [doc for doc in corpus if doc['id'] not in existing_ids]

        if new_docs:
            print(f"Found {len(new_docs)} new documents. Updating Vector DB...")
            ids = [d['id'] for d in new_docs]
            contents = [f"{d['title']}\n{d['content']}" for d in new_docs]
            metadatas = [
                {
                    "category": d.get('category', 'Unknown'), 
                    "error_code": d.get('error_code', 'N/A')
                } for d in new_docs
            ]
            
            self.vec_eng.collection.add(
                ids=ids,
                documents=contents,
                metadatas=metadatas
            )
            
            # If data changed, we must rebuild the BM25 index to maintain accuracy
            self.rebuild_keyword_index(corpus)
        else:
            print("No new documents detected in JSON.")
            # If no new data, try to load the BM25 index from disk
            if not self.load_index():
                print("BM25 index not found on disk. Building from scratch...")
                self.rebuild_keyword_index(corpus)
            else:
                print("BM25 index loaded successfully from disk.")

    def rebuild_keyword_index(self, corpus):
        """
        Rebuilds the BM25 index from the entire corpus and saves it to disk.
        
        Args:
            corpus (list): The complete list of documents.
        """
        print("Rebuilding BM25 Keyword Index...")
        self.kw_eng.from_corpus(corpus)
        self.save_index()

    def save_index(self):
        """Serializes the KeywordEngine state to a pickle file."""
        try:
            with open(self.bm25_loc, 'wb') as f:
                # We save the whole kw_eng object which contains the BM25 matrix and IDs
                pickle.dump(self.kw_eng, f)
            print(f"BM25 index saved to {self.bm25_loc}")
        except Exception as e:
            print(f"Error saving BM25 index: {e}")

    def load_index(self):
        """
        Deserializes the KeywordEngine state from a pickle file.
        
        Returns:
            bool: True if loaded successfully, False otherwise.
        """
        if not self.bm25_loc.exists():
            return False
            
        try:
            with open(self.bm25_loc, 'rb') as f:
                loaded_kw_eng = pickle.load(f)
                # Transfer the loaded state to the current engine
                self.kw_eng.bm25 = loaded_kw_eng.bm25
                self.kw_eng.doc_ids = loaded_kw_eng.doc_ids
            return True
        except Exception as e:
            print(f"Error loading BM25 index: {e}")
            return False
