import logging
from typing import List, Dict, Any

def search_runbooks(query: str, collection: Any) -> str:
    """
    Searches the vector database for troubleshooting runbooks matching the query.
    Returns the top relevant runbook segments as a string.
    """
    logging.info(f"Searching runbooks for: {query}")
    
    try:
        # Perform semantic search
        results = collection.query(
            query_texts=[query],
            n_results=2,
            include=["documents", "metadatas"]
        )
        
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        
        if not documents:
            return "No relevant runbooks found in the knowledge base."
            
        formatted_results = []
        for doc, meta in zip(documents, metadatas):
            source = meta.get("filename", "Unknown Source")
            formatted_results.append(f"--- Source: {source} ---\n{doc}")
            
        return "\n\n".join(formatted_results)
        
    except Exception as e:
        logging.error(f"Knowledge search failed: {e}")
        return f"Error during runbook retrieval: {str(e)}"
