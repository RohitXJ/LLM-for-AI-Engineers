import os
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
import chromadb
from dotenv import load_dotenv

load_dotenv()

# -----------------------------------------------------
# 1. THE THEORY: THE SELF-QUERYING PATTERN
# -----------------------------------------------------


class SearchFilters(BaseModel):
    """The structured filter generated from a natural language query."""
    topic: Optional[str] = Field(None, description="Filter by topic (e.g., Security, IT, HR)")
    year: Optional[int] = Field(None, description="Filter by a specific year")
    priority: Optional[str] = Field(None, description="Filter by priority: High, Medium, Low")

# Initialize the "Translator" LLM
# Using gemini-2.5-flash-lite for optimal cost/performance
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
structured_llm = llm.with_structured_output(SearchFilters)

def translate_query_to_filter(user_query: str) -> Dict[str, Any]:
    """Translates natural language to a ChromaDB-compatible filter dict."""
    print(f"\n[User Query]: \"{user_query}\"")
    
    # Prompt the LLM to extract filters
    extracted = structured_llm.invoke(f"Extract filters from this search query: {user_query}")
    
    # Convert Pydantic model to ChromaDB 'where' clause
    filters = {}
    items = extracted.model_dump(exclude_none=True)
    
    if len(items) == 1:
        # Simple single filter: {"key": "value"}
        filters = items
    elif len(items) > 1:
        # Multiple filters: {"$and": [{"key1": "val1"}, {"key2": "val2"}]}
        filters = {"$and": [{k: v} for k, v in items.items()]}
        
    return filters

# -----------------------------------------------------
# 2. RUNNING THE SELF-QUERYING LOOP
# -----------------------------------------------------

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="automated_metadata_demo")

def self_query(user_query: str):
    # Step 1: Translate NL to Filter
    where_filter = translate_query_to_filter(user_query)
    print(f"[Generated Filter]: {where_filter}")
    
    # Step 2: Semantic Search + Metadata Filtering
    results = collection.query(
        query_texts=[user_query], # We still use the text for vector search
        n_results=2,
        where=where_filter if where_filter else None
    )
    
    if not results["documents"][0]:
        print("AI-> No documents found matching those specific filters.")
    else:
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            print(f"AI-> Found: {doc} | Metadata: {meta}")

# --- Test Scenarios ---
# Scenario A: Specific filters extracted from text
self_query("Find me IT documents from 2023")

# Scenario B: Implicit filters
self_query("What wellness programs do we have for 2024?")

# Scenario C: Broad query (should result in no filter)
self_query("Tell me about everything in the database")
