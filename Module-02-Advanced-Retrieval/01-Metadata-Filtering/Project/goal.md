# Project Goal: OmniSearch - Metadata-Driven Research Engine

## 🎯 The Vision
Standard RAG systems often fail when a user asks for something specific like "Show me *beginner* level security docs from *2023*." Vector search alone might return an "advanced" doc from 2024 because the semantic meaning is similar.

**OmniSearch** solves this by implementing a **Senior-Level Metadata Pipeline**. It uses LLMs to automatically "tag" incoming data and translate user intent into structured database filters.

---

## 🏗️ Technical Architecture
The project is built on three pillars:
1.  **Automated Extraction:** No manual tagging. Gemini 2.0 Flash Lite analyzes every chunk to extract a rich metadata schema.
2.  **Self-Querying Retrieval:** The system "listens" to the user's query and builds a ChromaDB-compatible logical filter (e.g., using `$and`, `$gte`, `$in`).
3.  **Transparent Retrieval:** The UI explains *why* a result was found by showing the applied filters.

---

## 📥 Inputs vs. 📤 Outputs

### 1. Ingestion Phase
*   **Input:** Raw Text files (.txt) or PDFs containing research/corporate data.
*   **Process:** 
    *   Text is chunked (Recursive Character).
    *   LLM extracts: `topic`, `year`, `complexity` (Beginner/Intermediate/Advanced), and `audience`.
*   **Output:** Vector Embeddings + JSON Metadata stored in **ChromaDB**.

### 2. Query Phase (Self-Querying)
*   **Input:** Natural Language Query (e.g., *"What were the high-priority security updates in 2024 for IT?"*)
*   **Process:**
    *   LLM generates a `filter_dict`: `{"$and": [{"topic": "Security"}, {"year": 2024}, {"priority": "High"}]}`.
    *   ChromaDB executes a metadata-filtered semantic search.
*   **Output:** 
    *   **The Answer:** Relevant document snippets.
    *   **The Logic:** Visualization of the generated filter used to find the data.

---

## 🛠️ Tech Stack
*   **LLM:** Google Gemini 2.0 Flash Lite (for Extraction & Query Translation).
*   **Vector DB:** ChromaDB (with Nested Logic support).
*   **Embedding Model:** `all-MiniLM-L6-v2` (Local, cost-efficient).
*   **Validation:** Pydantic (to ensure LLM outputs are software-ready).
*   **Frontend:** Streamlit.

---

## 🚀 Mastery Checklist
- [ ] Implement Automated Metadata Extraction via LLM.
- [ ] Support Nested Logical Filters (`$and`, `$or`).
- [ ] Build a "Query Translator" (Natural Language -> Metadata Filter).
- [ ] Create a Streamlit Dashboard with "Retrieval Reasoning" views.
- [ ] Verify with RAGAS-style precision checks (Manual/LLM-as-a-Judge).
