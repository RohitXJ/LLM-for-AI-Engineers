import chromadb, logging
from ollama import Client
from pathlib import Path

class Manager():
    def __init__(self, logger:logging.Logger, MODEL:str = "gpt-oss:120b-cloud", RUNBOOKS:Path = Path(r"runbooks")):
        self.logger = logger
        self.logger.info("Initializing Manager and ChromaDB...")
        self.chroma = chromadb.PersistentClient("./database/chroma_data")
        self.collection = self.chroma.get_or_create_collection(name="runBooks")
        self.logger.info("ChromaDB collection 'runBooks' ready.")
        self.client = Client(host="http://localhost:11434")
        self.MODEL = MODEL
        self.RUNBOOKS = RUNBOOKS

    def engineStartup(self):
        from core.database.vector_store import vectorDB
        self.vdb = vectorDB(self.chroma, self.collection, self.RUNBOOKS, self.logger)
        new_files = self.vdb.checkNewFiles()
        print(f"Files found {new_files}")
        try:
            if new_files:
                self.logger.info("New Runbooks found, Initializing Ingestion Pipeline")
                try:
                    self.vdb.ingestNewFiles(new_files)
                    self.logger.info("Ingestion Pipeline Completed Successfully")
                except Exception as e:
                    self.logger.error(f"Ingestion Pipeline Disrupted: {e}")
            else:
                self.logger.info("No new Runbooks found!")
            self.logger.info(f"Engine Startup Successful")
        except Exception as e:
            self.logger.error(f"Engine Startup Unsuccessful : {e}")
                