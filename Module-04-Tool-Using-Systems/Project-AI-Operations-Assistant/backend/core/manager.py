import chromadb, logging
from ollama import Client
from pathlib import Path

class Manager():
    def __init__(self, logger:logging.Logger, MODEL:str = "gpt-oss:120b-cloud", RUNBOOKS:Path = Path(r"runbooks")):
        self.chroma = chromadb.PersistentClient("./database/chroma_data")
        self.collection = chromadb.Collection("runBooks")
        self.client = Client(host="http://localhost:11434")
        self.MODEL = MODEL
        self.RUNBOOKS = RUNBOOKS
        self.logger = logger

    def engineStartup(self):
        from core.database.vector_store import vectorDB
        self.vdb = vectorDB(self.chroma, self.collection, self.RUNBOOKS)
        new_files = self.vdb.checkNewFiles()
        if new_files:
            pass
        else:
            pass