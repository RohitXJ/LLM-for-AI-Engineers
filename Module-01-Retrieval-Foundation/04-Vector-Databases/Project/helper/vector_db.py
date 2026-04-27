import chromadb
from chromadb.utils import embedding_functions

def chroma_init(collection_name="knowledge_base"):
    """
    Initializes an in-memory (ephemeral) ChromaDB client with default embedding function.
    """
    client = chromadb.EphemeralClient()
    
    # Use Chroma's default embedding function (all-MiniLM-L6-v2)
    # This avoids manual embedding in the main loop
    default_ef = embedding_functions.DefaultEmbeddingFunction()
    
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=default_ef
    )
    return collection
