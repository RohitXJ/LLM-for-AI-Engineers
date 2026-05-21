# Project Goal: The "Query Intelligence" RAG Engine

## 🎯 Overview
Build a production-grade RAG pipeline that transforms "messy" human queries into high-precision search operations. This project moves beyond simple vector search by implementing a multi-stage **Query Transformation & Retrieval Orchestrator**.

## 🏗️ Production Architecture (The "Super-Retriever" Pipeline)

### 1. The Input & Transformation Layer
*   **Query Refining (Anaphora Resolution):** Convert conversational follow-ups (e.g., "What about its cost?") into standalone queries using chat history.
*   **Parallel Transformation Path:**
    *   **Decomposition:** If the query is complex, split it into atomic sub-questions.
    *   **Multi-Query (Query Expansion):** Generate 3 variations to handle synonym/vocabulary mismatch.
*   *Note: HyDE is explicitly excluded to prioritize precision over hallucinated overlap.*

### 2. The Retrieval Layer (Hybrid + Filtered)
*   **Metadata Filtering:** Narrow down the search space using pre-defined categories/tags.
*   **Hybrid Search (BM25 + Vector):** 
    *   **Keyword (BM25):** Catch specific terminology and IDs.
    *   **Semantic (Vector):** Catch the general intent.
*   **Reciprocal Rank Fusion (RRF):** Intelligently merge results from keyword and semantic paths.

### 3. The Post-Processing Layer (The "Truth" Filter)
*   **Re-ranking:** Use a Cross-Encoder (e.g., BGE-Reranker) to score the relationship between the *original* query and the retrieved chunks. This eliminates "vector noise."
*   **Context Compression:** Select only the top-K chunks that fit the LLM's context window while maximizing information density.

### 4. Generation
*   **Grounded Response:** Generate the final answer using the highly-refined context set.

## 🛠️ Technical Stack
*   **LLM:** Ollama (for transformation) & Google Gemini (for final generation - optional).
*   **Vector DB:** ChromaDB (Persistent mode).
*   **Search:** `rank_bm25` for keyword search.
*   **Re-ranking:** `sentence-transformers` (Cross-Encoders).
*   **Framework:** Modular Python (no heavy frameworks like LangChain, to keep logic transparent).

## ✅ Success Metrics (RAGAS Focus)
*   **Faithfulness:** Does the answer come strictly from the context?
*   **Answer Relevance:** Does the answer actually address the user's intent?
*   **Context Recall:** Did we find *all* the relevant information (tested via Decomposition)?

## 📂 Project Structure
```text
Project/
├── main.py              # CLI Entry Point
├── core/
│   ├── transformer.py   # Refining, Decomposition, Multi-Query
│   ├── retriever.py     # Hybrid Search & Metadata logic
│   ├── reranker.py      # Cross-Encoder scoring
│   └── database.py      # ChromaDB initialization & management
├── data/                # Sample knowledge base (JSON/Text)
└── goal.md              # This file
```
