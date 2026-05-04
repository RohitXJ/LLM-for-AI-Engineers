import json
from core import Chroma_VDB

COLLECTION_NAME = "SMART_DOC_CHAT"
VDB_PATH = "./chroma_db"

def show_unique_metadata():
    """
    Queries ChromaDB to extract and display all unique metadata values 
    present in the current collection.
    """
    print(f"\n--- 🔍 Inspecting Metadata Diversity [{COLLECTION_NAME}] ---\n")
    
    try:
        # Instantiate the database wrapper
        chroma = Chroma_VDB(DB_PATH=VDB_PATH, collection_name=COLLECTION_NAME)
        
        # Fields we want to inspect
        fields = ["topic", "year", "complexity", "priority", "audience"]
        
        stats = {}
        for field in fields:
            unique_values = chroma.get_unique_values(field)
            # Sort for better readability if possible
            try:
                unique_values.sort()
            except:
                pass
            stats[field] = unique_values

        # Pretty print the results
        print(json.dumps(stats, indent=4))
        
        # Summary stats
        total_chunks = len(chroma.collection.get()['ids'])
        print(f"\n--- Total Indexed Chunks: {total_chunks} ---")
        print("--- Inspection Complete ---\n")

    except Exception as e:
        print(f"❌ Error accessing database: {e}")

if __name__ == "__main__":
    show_unique_metadata()
