import os
import json
from typing import List, Optional
from pydantic import BaseModel, Field
import chromadb
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load API Keys
load_dotenv()

# -----------------------------------------------------
# 1. THE ARCHITECTURE: STRUCTURED EXTRACTION
# -----------------------------------------------------
# THEORY: A Senior AI Engineer doesn't manually tag data. 
# We use a "Metadata Extractor" (LLM) to analyze raw text 
# and output a structured schema (JSON) before ingestion.

class DocumentMetadata(BaseModel):
    """Schema for the metadata we want to extract from every document."""
    topic: str = Field(description="The primary subject (e.g., Security, Finance, HR)")
    priority: str = Field(description="Priority level: High, Medium, or Low")
    entities: List[str] = Field(description="List of key names, departments, or systems mentioned")
    year: int = Field(description="The year mentioned in the text, default to 2024 if not found")

# Initialize the LLM (Gemini)
# Using gemini-2.5-flash-lite - The most cost-efficient model available for you
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

def extract_metadata(text: str) -> dict:
    """Uses LLM to turn raw text into structured metadata dictionary."""
    print(f"--> Extracting metadata for: {text[:50]}...")
    
    # We use the LLM with structured output capabilities
    structured_llm = llm.with_structured_output(DocumentMetadata)
    
    try:
        metadata_obj = structured_llm.invoke(f"Extract metadata from this text: {text}")
        # SENIOR FIX: ChromaDB DOES NOT support lists in metadata. 
        # We must convert the list of entities into a comma-separated string.
        data = metadata_obj.model_dump()
        data["entities"] = ", ".join(data["entities"]) if data["entities"] else "None"
        return data
    except Exception as e:
        print(f"Error extracting metadata: {e}")
        # Fallback must also follow the "no-list" rule
        return {"topic": "Unknown", "priority": "Low", "entities": "None", "year": 2024}

# -----------------------------------------------------
# 2. INGESTION WITH AUTOMATED TAGS
# -----------------------------------------------------

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="automated_metadata_demo")

raw_docs = [
    "The 2023 server migration was handled by the IT department. Priority was critical due to data risks.",
    "New employee wellness program starting in 2024. HR will oversee the rollout for all remote staff.",
    "Quarterly financial audit for 2023 showed a 10% increase in operational costs. Finance team to review."
]

print("\n--- Starting Automated Ingestion ---")
for i, text in enumerate(raw_docs):
    # STEP 1: Extract Metadata using LLM
    metadata = extract_metadata(text)
    
    # STEP 2: Add to Vector DB
    collection.add(
        ids=[f"doc_{i}"],
        documents=[text],
        metadatas=[metadata]
    )
    print(f"AI Tagged -> {metadata}")

# -----------------------------------------------------
# 3. VERIFICATION
# -----------------------------------------------------
print("\n--- Testing Metadata-Aware Query ---")

# Let's find documents from 2023 that are High/Critical priority
results = collection.query(
    query_texts=["server and audit issues"],
    where={"$and": [
        {"year": 2023},
        {"priority": {"$in": ["High", "Critical", "critical"]}} # LLMs might vary in casing
    ]},
    n_results=5
)

for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"\n[Found]: {doc}")
    print(f"[Metadata]: {meta}")
