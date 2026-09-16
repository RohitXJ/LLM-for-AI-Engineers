import chromadb
from pathlib import Path
class vectorDB():
    def __init__(self, chroma:chromadb.PersistentClient, collection:chromadb.Collection, targetDir:Path):
        self.chroma = chroma
        self.collection = collection
        self.targetDir = targetDir

    def checkNewFiles(self):
        """Check for new files in the target directory that are not in the collection."""

        local_list = self.listFilesFromStorage()
        vdb_list = self.listFilesFromVDB()
        return list(set(local_list) - set(vdb_list))

    def listFilesFromStorage(self) -> list[str]:
        """Get all the filenames from the target directory that end with .json"""
    
        return [file.name for file in self.targetDir.glob("*.json") if file.is_file()]

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
