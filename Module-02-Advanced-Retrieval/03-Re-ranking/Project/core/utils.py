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