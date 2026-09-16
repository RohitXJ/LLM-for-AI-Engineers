from ollama import Client
import json, chromadb
from pathlib import Path
import logging
logging.basicConfig(
    filename='logs/ai.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
#Globals
main_logger = logging.getLogger("MainApp")
LOG_ADD = r"alerts"
#Imports
from core import Manager

def main():
    manager = Manager(main_logger)
    manager.engineStartup()

if __name__ == "__main__":
    main()