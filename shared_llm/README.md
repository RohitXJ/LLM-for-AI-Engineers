# 🚀 shared_llm: The Core AI Engineering Library

`shared_llm` is a centralized, production-grade library designed to power the RAG (Retrieval-Augmented Generation) projects in the **LLM-for-AI-Engineers** roadmap. It standardizes common patterns, reduces boilerplate, and ensures architectural consistency across all modules.

---

## 🏗️ Technical Architecture

The library is organized into specialized modules:

1.  **`schema.py`**: Pydantic models for data contracts (Metadata, Filters).
2.  **`database.py`**: A robust manager for ChromaDB operations.
3.  **`llm.py`**: Unified interface for chat, extraction, and self-querying.
4.  **`retrieval.py`**: Hybrid search engines (BM25 + RRF).
5.  **`reranking.py`**: High-precision Local (BGE) and Cloud (Cohere) rerankers.
6.  **`processing.py`**: Data utilities for chunking and loading.

---

## 🛠️ Installation

To use this library in any project within this workspace, perform an **Editable Install**:

```bash
pip install -e .
```

This allows you to modify the library code and have the changes reflected immediately in your projects without re-installing.

---

## 📖 Usage Examples

### 1. Unified LLM Chat & Extraction
```python
from shared_llm import ChatManager

ai = ChatManager(model="gpt-oss:20b-cloud")

# Ask a grounded question
answer = ai.ask(query="What is the policy?", context="...")

# Extract structured metadata
meta = ai.extract_metadata(text="...")
```

### 2. Hybrid Search (Keyword + Vector)
```python
from shared_llm import KeywordEngine, HybridFusion, ChromaManager

# 1. Search Vector DB
db = ChromaManager(path="./chroma_db")
v_results = db.query(query_text="...", n_results=10)
v_ids = v_results['ids'][0]

# 2. Search Keyword Engine
kw = KeywordEngine()
kw.load("./bm25.pkl")
kw_results = kw.search(query="...", top_k=10)

# 3. Fuse Rankings (RRF)
fused = HybridFusion.fuse(keyword_results=kw_results, vector_results=v_ids)
```

### 3. High-Precision Re-ranking
```python
from shared_llm import LocalReranker

reranker = LocalReranker()
top_results = reranker.rerank(
    query="Specific question?",
    documents=["Chunk A", "Chunk B"],
    top_n=1
)
```

---

## 💎 Engineering Standards
- **Strict Typing:** Every function uses Python type hints for better IDE support.
- **Google-Style Docstrings:** Detailed documentation for every parameter and return value.
- **Architectural Separation:** Logic is decoupled (e.g., the database doesn't know about the LLM).

---
*Built for the next generation of Senior AI Engineers.*
