# 🛡️ Sentinel Search: The Hybrid Intelligence Engine

![Sentinel Search Hero](./imgs/BASE%20UI.png)

## 🎯 The Problem: Why Naive RAG Fails
In a corporate environment, policy documents are often logically dense. A query like *"Can a Junior Dev request a Mac Studio after 6 months?"* might fail in a standard vector-only search because:
1. **Keyword Overlap**: Multiple documents mention "Junior Dev" or "Mac Studio."
2. **Semantic Nuance**: The "6-month" condition is a specific logical constraint that vector similarity often misses.
3. **Information Overload**: Retrieving too many documents (Top-K) dilutes the LLM's context window with irrelevant noise.

**Sentinel Search** solves this by moving beyond "Semantic Search" into a **Multi-Stage Retrieval Pipeline**.

---

## 🏗️ The 3-Stage Pipeline Architecture

To achieve gold-standard accuracy, we implement a sequential filtering and ranking system:

### 1️⃣ Metadata Pre-filtering
Before searching, we use the LLM to translate natural language into structured metadata filters (e.g., `department: "Engineering"`). This narrows the search space from thousands of documents to a relevant subset, drastically reducing "noise" and improving retrieval speed.

### 2️⃣ Hybrid Search (BM25 + Vector) with RRF
We combine the strengths of two worlds:
- **BM25 (Keyword)**: Perfect for catching specific acronyms, product names (like "Mac Studio"), and numeric values.
- **Vector (Semantic)**: Captures the general intent of the user.
- **Reciprocal Rank Fusion (RRF)**: Intelligently merges these results into a single ranked list.

### 3️⃣ Cross-Encoder Re-ranking (The "Brain")
The top 10-20 candidates from the hybrid stage are passed to a **Cross-Encoder**. Unlike standard encoders, a Cross-Encoder performs a deep, pair-wise semantic analysis of the query vs. the document. This is the "final check" that ensures the absolute most relevant document is ranked at #1.

---

## 🚀 Key Features

### 🌩️ Dual-Mode Intelligence (Local vs. Cloud)
Switch seamlessly between privacy and performance:
- **Local Ninja Mode**: Uses `BAAI/bge-reranker-base` locally. Fully private and air-gapped.
- **Cloud Beast Mode**: Integrates **Cohere Rerank v3** for industry-leading accuracy.
- *Switch via a single slider in the UI.*

### 📊 Real-Time Observability
We don't just give answers; we show the work:
- **Latency Tracking**: See exactly how long each of the 3 stages takes.
- **Re-rank Visualizer**: A UI component showing how the Cross-Encoder changed the initial ranking.
- **Context Preview**: See exactly what text is being sent to the LLM.

![System Stats](./imgs/Model%20Choose%20and%20System%20Stat(docs%20index%20and%20dpt.).png)

---

## ⏱️ Performance & Accuracy
Despite the complexity of running three levels of checking, Sentinel Search achieves **~3 seconds of total latency** (on average), delivering strictly grounded and correct answers.

![Query Interface](./imgs/Query%20Example_latency%20of%20each%20steps%20on%20right_filters%20used%20to%20the%20left.png)

---

## 🛠️ Tech Stack
- **Vector DB**: ChromaDB
- **Retrieval Engine**: `rank_bm25` + `SentenceTransformers`
- **Re-ranking**: `CrossEncoder` (Local) / `Cohere` (Cloud)
- **UI Framework**: Streamlit
- **LLM**: Ollama (Local)

---
*Developed by a Senior AI Engineer to prove that Accuracy is a choice, not a chance.*
