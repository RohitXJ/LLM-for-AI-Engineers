import os
import datetime
import time
from pathlib import Path
from PyPDF2 import PdfReader
import docx

def knowledge_chk(path: str) -> list:
    full_path = Path(path)
    if not full_path.exists():
        full_path.mkdir(parents=True, exist_ok=True)
        return []
    
    files = [str(f) for f in full_path.iterdir() if f.is_file()]
    return files

async def extract_text(file_path: str) -> list[dict]:
    """
    Extracts text and returns a list of 'Document' dictionaries.
    Each dict: {"content": str, "metadata": dict}
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # File-level metadata
    file_stat = path.stat()
    mod_date = datetime.datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
    
    base_metadata = {
        "source": path.name,
        "extension": ext,
        "last_modified": mod_date
    }

    documents = []

    try:
        if ext in [".txt", ".md"]:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                documents.append({
                    "content": content,
                    "metadata": {**base_metadata, "page": 1}
                })

        elif ext == ".pdf":
            reader = PdfReader(file_path)
            num_pages = len(reader.pages)
            for i, page in enumerate(reader.pages):
                content = page.extract_text()
                if content and content.strip():
                    documents.append({
                        "content": content,
                        "metadata": {**base_metadata, "page": i + 1}
                    })
            
            if not documents:
                print(f"Warning: No text could be extracted from {path.name}. It might be a scanned image or encrypted.")

        elif ext == ".docx":
            doc = docx.Document(file_path)
            pages = []
            current_page_text = []
            
            for para in doc.paragraphs:
                if 'lastRenderedPageBreak' in para._element.xml or 'w:br' in para._element.xml and 'type="page"' in para._element.xml:
                    if current_page_text:
                        pages.append("\n".join(current_page_text))
                    current_page_text = [para.text]
                else:
                    current_page_text.append(para.text)
            
            if current_page_text:
                pages.append("\n".join(current_page_text))

            for i, page_content in enumerate(pages):
                if page_content.strip():
                    documents.append({
                        "content": page_content,
                        "metadata": {**base_metadata, "page": i + 1}
                    })
        else:
            # Silently ignore unsupported files when batching, or raise for single file
            raise ValueError(f"Unsupported file extension: {ext}")

        return documents

    except Exception as e:
        raise Exception(f"Error processing {file_path}: {str(e)}")
    
import asyncio
import json

async def test_extraction():
    # 1. Provide a path to a real file on your machine
    test_path = r"Module-01-Retrieval-Foundation\04-Vector-Databases\Project\knowledge_base" 
    test_files = knowledge_chk(test_path)
    for test_file in test_files:
        try:
            print(f"--- Extracting: {test_file} ---")
            
            # 2. Await the async function
            raw_output = await extract_text(test_file)
            
            # 3. Print the raw list of dictionaries
            # Using json.dumps makes the metadata and content easier to read
            print(json.dumps(raw_output, indent=4))
            
            # Optional: Print just the first page's content to verify
            if raw_output:
                print("\n--- First Page Content Preview ---")
                print(raw_output[0]['content'][:500] + "...") 

        except Exception as e:
            print(f"Test Failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_extraction())