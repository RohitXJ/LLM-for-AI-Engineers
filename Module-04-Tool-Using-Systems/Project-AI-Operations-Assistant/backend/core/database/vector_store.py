import chromadb, os
from pathlib import Path
import logging
class vectorDB():
    def __init__(self, chroma:chromadb.PersistentClient, collection:chromadb.Collection, targetDir:Path, logger = logging.Logger):
        self.logger = logger
        self.logger.info("Initializing Vector DB")
        self.chroma = chroma
        self.collection = collection
        self.targetDir = targetDir
        self.logger.info("Initializing Vector DB Successful")

    def checkNewFiles(self):
        """Check for new files in the target directory that are not in the collection."""

        self.logger.info("Initializing New Runbooks search")
        local_list = self.listFilesFromStorage()
        vdb_list = self.listFilesFromVDB()
        return list(set(local_list) - set(vdb_list))

    def listFilesFromStorage(self) -> list[str]:
        """Get all the filenames from the target directory that end with .json"""
    
        return [file.name for file in self.targetDir.glob("*.md") if file.is_file()]

    def listFilesFromVDB(self) -> list[str]:
        """Get all unique filenames from the collection safely handling pagination."""

        unique_files = set()
        limit = 100
        offset = 0

        while True:
            results = self.collection.get(include=["metadatas"], limit=limit, offset=offset)
            metadatas = results.get("metadatas", [])

            if not metadatas:
                break

            # Extract filenames safely and add to our set
            for meta in metadatas:
                if meta and "filename" in meta:
                    unique_files.add(meta["filename"])

            # Move to the next page
            offset += limit

        return list(unique_files)

    def ingestNewFiles(self, new_files: list[str]):
        """Read text from new files, split them if necessary, and add them to the collection."""

        for filename in new_files:
            file_path = self.targetDir / filename
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Skip empty files
                if not content.strip():
                    continue
                
                # Placeholder for document chunker
                doc_id = f"{filename}_0"
                
                self.collection.add(
                    documents=[content],
                    metadatas=[{"filename": filename}],
                    ids=[doc_id]
                )
                self.logger.info(f"Ingested {filename} to {self.collection.name}")
                
            except Exception as e:
                self.logger.error(f"Error ingesting file {filename}: {e}")

        
