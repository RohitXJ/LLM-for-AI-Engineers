# 🧠 Local Knowledge Explorer (Naive RAG)

A production-ready CLI tool that transforms local text and markdown files into a searchable, persistent knowledge base. This project demonstrates the foundational **Naive RAG (Retrieval-Augmented Generation)** architecture, featuring semantic chunking and grounded LLM response logic.

---

## 🚀 Features

- **Hybrid Chunking:** Combines `SemanticChunker` (meaning-based breakpoints) with `RecursiveCharacterTextSplitter` (size-constrained refinement) for optimal context delivery.
- **Persistent Vector Store:** Uses **ChromaDB** to ensure your knowledge base persists across sessions.
- **Relevance Guard:** Implements a strict distance threshold (1.2) to prevent the LLM from processing irrelevant data, reducing hallucinations and API costs.
- **Async Ingestion Pipeline:** High-performance file processing using `asyncio` and semaphores for concurrent data dumping.
- **Strict Grounding:** A specialized system prompt ensures the AI never uses external knowledge, citing only the provided sources.

---

## 🏗️ Architecture

1.  **Ingest:** Scans `data/` directory for `.txt` and `.md` files.
2.  **Chunk:** Split documents semantically then refine to ~500 character blocks.
3.  **Embed:** Generate vectors using `all-MiniLM-L6-v2`.
4.  **Store:** Save vectors and metadata (source name, chunk index) in ChromaDB.
5.  **Retrieve:** Query top-3 chunks and filter via Distance Threshold.
6.  **Augment:** Feed filtered context into `gpt-oss:20b-cloud` (via Ollama).

---

## 🛠️ Setup & Usage

### 1. Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running.
- Required model pulled: `ollama pull gpt-oss:20b-cloud` (or update `CHAT_MODEL` in code).

### 2. Installation
```bash
pip install -r requirements.txt
```

### 3. Running the Engine

**Standard Ingestion & Chat:**
```bash
python main.py -fd ./data
```

**High-Speed Async Ingestion:**
```bash
python async_v_main.py -fd ./data
```

**Query Mode (Skip Ingestion):**
```bash
python main.py
```

---

## 📂 Project Structure
- `main.py`: The core synchronous RAG application.
- `async_v_main.py`: High-performance asynchronous version for batch ingestion.
- `chroma_db/`: Local directory where the persistent vector database is stored.
- `data/`: The target directory for your source documents.

---

## 🎓 Senior AI Engineer Insights

### Why Semantic + Recursive Chunking?
Standard character splitting often cuts sentences in half. Semantic splitting finds natural breaks in thought, while Recursive refinement ensures those thoughts fit within the LLM's context window without overflow.

### The Importance of Distance Thresholds
LLMs are "helpful" by nature and will try to answer even if the retrieved context is garbage. By blocking irrelevant chunks at the database level (Distance > 1.2), we ensure the model only receives high-signal information.

---
*Created as part of the LLM-for-AI-Engineers Master Roadmap.*
