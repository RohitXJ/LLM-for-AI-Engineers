from ollama import Client
import json
from pathlib import Path
import logging
logging.basicConfig(
    filename='logs/ai.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

LOG_ADD = r"alerts"

def readLogs(log_path: str) -> list[dict]:
    """
    Reads logs from a directory and returns a list of dictionaries.
    Each dictionary contains log data and the name of the log file.
    """

    logging.info("Log Reader Called")
    path = Path(log_path)
    logs_list = []
    
    for f in path.iterdir():
        if f.is_file():
            try:
                with open(f, 'r', encoding='utf-8') as file:
                    file_content = json.load(file)
                    
                    # Handle if a file contains a list of dicts
                    if isinstance(file_content, list):
                        for log_dict in file_content:
                            if isinstance(log_dict, dict):
                                log_dict["log_file_name"] = f.name
                        
                        logs_list.extend(file_content)
                        
                    # Handle if a file contains just a single dict
                    elif isinstance(file_content, dict):
                        file_content["log_file_name"] = f.name
                        logs_list.append(file_content)
                        
                    logging.info(f"Extraction successful for '{f}'")       
            except Exception as e:
                logging.error(f"Failed to read {f.name}: {e}")
                
    return logs_list

def main():
    #Read Logs
    logs_list = readLogs(LOG_ADD)
    print(logs_list)

if __name__ == "__main__":
    main()