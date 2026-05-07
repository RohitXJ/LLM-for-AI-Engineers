# 🛡️ Sentinel Search: The Hybrid Intelligence Engine

## 🎯 The Vision
Build a production-grade retrieval engine capable of handling complex, logic-heavy corporate documents (Policies, Legal, SRE playbooks). The system must prove that "Semantic Search" alone is not enough and that **Hybrid Search + Re-ranking** is the gold standard for accuracy.

## 🚀 Key Features

### 1. Dual-Engine Architecture
*   **🌩️ Cloud Beast Mode:** High-performance mode using OpenAI Embeddings and Cohere Re-ranker.
*   **🥷 Local Ninja Mode:** Fully private, air-gapped mode using local `all-MiniLM-L6-v2` for embeddings and `BAAI/bge-reranker-base` for re-ranking.
*   *Switch seamlessly between them via a single toggle.*

### 2. Multi-Stage Retrieval Pipeline
1.  **Metadata Pre-filtering:** Narrow down results by Category, Department, or Year before searching.
2.  **Hybrid Search (The Backbone):** 
    *   **BM25 (Keyword):** To catch specific error codes, acronyms, and product names.
    *   **Vector (Semantic):** To catch general intent and meaning.
    *   **Reciprocal Rank Fusion (RRF):** To merge these results intelligently.
3.  **Cross-Encoder Re-ranking (The Brain):** Take the top 10-20 hybrid results and perform deep semantic re-ranking to find the absolute truth.

### 3. Visual Observability (The "Why")
*   **Rerank Visualizer:** A UI component showing the "Before vs After" ranking positions.
*   **Latency Metrics:** Transparent tracking of how much time each stage (Retrieval vs Reranking) takes.
*   **Score Breakdown:** Displaying BM25 vs Vector vs Re-ranker scores.

## 🛠️ Tech Stack
*   **Vector DB:** ChromaDB (Local Persistence).
*   **Retrieval:** `rank_bm25` (Keyword) + `SentenceTransformers` (Local Vector).
*   **Re-ranking:** `CrossEncoder` (Local) / `Cohere` (Cloud API).
*   **UI:** Streamlit (Professional Dashboard).
*   **Logic:** Custom "Search Manager" class using the Adapter Pattern.

## 📂 Data Strategy
We will use a specialized dataset of **"Enterprise Policy Documents"** where subtle logic matters:
*   *Example Query:* "Can a Junior Dev request a Mac Studio after 6 months?"
*   *Challenge:* Multiple docs might mention "Mac Studio" or "Junior Dev," but only one specific policy covers the "6-month" condition.

## ✅ Success Metrics
*   **Accuracy:** The system must correctly rank the most logical answer as #1.
*   **Reliability:** Seamless switching between Local and Cloud modes.
*   **Performance:** Local re-ranking must stay under 1.5 seconds for the top 20 docs.

---
*Created by the Senior AI Engineer for the "Sentinel Search" Masterclass.*
