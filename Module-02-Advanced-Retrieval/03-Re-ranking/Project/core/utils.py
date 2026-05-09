import json
from pathlib import Path
def load_json(data_loc):
        """
        Reads all JSON files from the data directory and aggregates them.
        
        Returns:
            list: A list of all documents found in the JSON files.
        """
        all_documents = []
        data_path = Path(data_loc)
        
        if not data_path.exists():
            print(f"Warning: Directory {data_loc} not found.")
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

def get_ids(context: list[dict]):
    return context['ids'][0]

def filter_context_by_search(context, search_results):
    """
    Extracts documents and metadata from the context variable 
    using the ranked IDs from search_results.
    
    Args:
        context (dict): The original multi-nested query payload from Chroma.
        search_results (list): A list of tuples, e.g., [(doc_id, score), ...]
        
    Returns:
        dict: A clean map of doc_id -> {"document": str, "metadata": dict}
              preserving the rank order of search_results.
    """
    if not search_results or not context:
        return {}

    # 1. Flatten Chroma's nested list-of-lists format into a simple lookup map
    flat_context = {}
    for sub_ids, sub_docs, sub_metas in zip(context['ids'], context['documents'], context['metadatas']):
        for doc_id, text, meta in zip(sub_ids, sub_docs, sub_metas):
            flat_context[doc_id] = {
                "document": text,
                "metadata": meta
            }

    # 2. Extract data using search_results to strictly preserve the ranked order
    id_wise_data = {}
    for doc_id, _ in search_results:
        if doc_id in flat_context:
            id_wise_data[doc_id] = flat_context[doc_id]

    return id_wise_data