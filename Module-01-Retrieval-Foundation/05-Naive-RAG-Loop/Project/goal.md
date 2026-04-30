# 🏆 Module 01 Capstone: Local Knowledge Explorer

## 🎯 Goal
Build a production-ready CLI tool that transforms a directory of raw text/markdown files into a searchable, persistent knowledge base using the Naive RAG architecture. This project synthesizes everything learned in Module 01 (Tokenization, Chunking, Embeddings, and Vector DBs).

---

## 🏗️ Core Requirements

### 1. Data Ingestion Pipeline
*   **Source:** Automatically scan a local `data/` directory for `.txt` and `.md` files.
*   **Chunking:** Implement **Recursive Character Text Splitting** (Target: 500 characters with 50-character overlap).
*   **Metadata:** Capture the **filename** and **chunk index** for every piece of text to enable source attribution.

### 2. Vector Storage (Persistence)
*   **Database:** Use **ChromaDB** with a **Persistent Client**.
*   **Embedding Model:** Utilize the **Default Embedding Function** (`all-MiniLM-L6-v2`) for efficiency.
*   **State Management:** The script must detect if a file has already been ingested to avoid duplicate chunks.

### 3. Retrieval Logic
*   **Semantic Search:** Retrieve the top **3 most relevant** chunks for any user query.
*   **Relevance Guard:** Implement a **Distance Threshold**. If no results are close enough, inform the user rather than feeding garbage to the LLM.

### 4. Generation & Citations
*   **Model:** Use `qwen-no-think` (or a similar lightweight, direct model).
*   **Grounding:** Strict system prompt to ensure the LLM *only* uses provided context.
*   **Citations:** The final answer must explicitly state which files were used (e.g., "Source: architecture.md").

---

## 🛠️ Success Criteria
1.  **Persistence:** I can close the script, reopen it, and query my data without re-ingesting.
2.  **Accuracy:** When asked about a specific fact in a file, the AI correctly identifies the file.
3.  **Robustness:** When asked about something *not* in the data, the AI refuses to answer (no hallucinations).

---

## 📂 Project Structure
```text
Module-01-Retrieval-Foundation/05-Naive-RAG-Loop/Project/
├── data/               # Drop your .txt/.md files here
├── chroma_db/          # Persistent database storage
├── main.py             # The core RAG application
└── README.md           # Instructions on how to use it
```

---
*Created for the journey to Senior AI Engineer. Master the foundation, then build the future.*
