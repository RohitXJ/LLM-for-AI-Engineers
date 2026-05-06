# Project Goal: SRE-Pulse - Advanced Hybrid Troubleshooting Engine 🛠️

**SRE-Pulse** is a professional-grade retrieval system designed to solve the "Ranking Problem" in technical documentation. It demonstrates mastery of **Hybrid Search** by combining the precision of keyword-based retrieval (BM25) with the semantic depth of vector-based retrieval (Dense Embeddings), fused via **Reciprocal Rank Fusion (RRF)**.

---

## 🏗️ Phase-by-Phase Implementation Plan

### Phase 1: Bimodal Data Engineering
**Goal:** Create a "Search-Hard" dataset that forces a failure in single-mode retrieval.
- **Task:** Construct a `sre_docs.json` containing 20+ technical entries.
- **Input:** None (Creative technical writing).
- **Process:** Write entries with "Trap Data":
    - *Keyword Traps:* Vague descriptions of latency/errors without using the word "latency".
    - *Vector Traps:* Specific alphanumeric error codes (e.g., `0x8004210B`) that look similar to other codes in vector space.
- **Output:** `data/sre_docs.json`.

### Phase 2: Ingestion Orchestration & Persistence (Senior-Level Optimization)
**Goal:** Ensure the system scales and starts up instantly.
- **Discovery:** Re-tokenizing JSON and re-ingesting vectors on every run is a performance bottleneck.
- **Task:** Build the `IngestionManager` in `core/manager.py`.
- **Process:**
    - **Chroma Optimization:** Perform a batch `collection.get(ids=...)` check to skip already-indexed documents.
    - **BM25 Persistence:** Since `rank_bm25` is in-memory only, we will use **Pickle** to serialize the object to `bm25_index.pkl`.
    - **Sync Logic:** Rebuild the BM25 index *only* if new data is detected during the sync check.
- **Output:** Persistent `chroma_db/` and `bm25_index.pkl`.

### Phase 3: Modular Engine Development
**Goal:** Build isolated, testable search components.
- **Task:** Implement `KeywordEngine` and `VectorEngine` classes.
- **Input:** `data/sre_docs.json`.
- **Process:** 
    - **KeywordEngine:** Implement BM25 using `rank_bm25`. Handle tokenization and stop-word removal.
    - **VectorEngine:** Implement ChromaDB storage with `all-MiniLM-L6-v2` embeddings.
- **Output:** `core/engines.py` containing both classes with standardized `search()` methods.

### Phase 3: RRF Fusion Logic
**Goal:** Implement the "Decision Brain" that merges disparate rankings.
- **Task:** Create the `HybridRetriever` that coordinates the two engines.
- **Input:** Ranked lists (IDs + Ranks) from Phase 2.
- **Process:** Apply the Reciprocal Rank Fusion (RRF) formula: $Score = \sum \frac{1}{k + rank}$.
- **Output:** `core/fusion.py` capable of returning a single, optimized ranked list.

### Phase 4: Search Quality Dashboard (The "Senior" Portfolio Piece)
**Goal:** Build a UI that proves the "Hybrid Advantage" to stakeholders/employers.
- **Task:** Create a Streamlit interface with side-by-side comparisons.
- **Input:** `HybridRetriever` logic.
- **Process:**
    - Develop a 3-column layout: **BM25 Only** vs **Vector Only** vs **Hybrid (RRF)**.
    - Add a "Logic Inspector" that shows the RRF math for the top result.
    - Implement a "Search Performance" table comparing top-1 relevance across different query types (Code-based vs Symptom-based).
- **Output:** `main.py` (Streamlit App).

---

## 📊 Data Schema (sre_docs.json)
```json
[
  {
    "id": "err_001",
    "title": "Database Connection Handshake Failure",
    "error_code": "0xDB_TIMEOUT_408",
    "content": "The application hangs during the initial TLS handshake with the PostgreSQL cluster. It usually happens when the network latency exceeds 500ms.",
    "solution": "Verify the pg_hba.conf settings and ensure the load balancer is not dropping idle connections.",
    "category": "Database"
  }
]
```

---

## 🚀 Mastery Checklist
- [ ] Construct a technical dataset with specific "retrieval traps."
- [ ] Implement BM25 with custom technical tokenization.
- [ ] Integrate ChromaDB for local vector search.
- [ ] Build a robust Reciprocal Rank Fusion (RRF) algorithm.
- [ ] Deliver a professional Streamlit dashboard with "Explainable Retrieval" views.

---
**Module:** Module 02 - Advanced Retrieval (Hybrid Search)
**Role:** Senior AI Engineer Portfolio Project
