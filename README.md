# 🚀 LLM-for-AI-Engineers: Senior AI Engineer Roadmap

Welcome to the specialized journey of transitioning from a prompt wrapper developer to a **Senior AI Engineer**. This repository follows an "Atomic Learning" philosophy, focusing on production-grade RAG, Agentic Workflows, and LLM Evaluation.

---

## 🏗️ Architectural Vision
This repository is designed to prove **Accuracy, Logic, and Reliability**. Every module is built to demonstrate not just *that* it works, but *why* it was designed that way, backed by metrics.

### 🗺️ The Atomic Roadmap

#### 📂 [Module 01: The Retrieval Foundation](./Module-01-Retrieval-Foundation/)
*Mastering how text becomes searchable "knowledge".*
- **[💸 The LLM Budgeting Tool](./Module-01-Retrieval-Foundation/01-Tokenization-Context-Windows/Project/)**
  - *Focus:* Token auditing, cost projection, and context window visualization.
- **[🏗️ CHUNKER-PRO CLI](./Module-01-Retrieval-Foundation/02-Atomic-Chunking-Strategies/Project/)**
  - *Focus:* Multi-strategy text splitting (Recursive, Token, Semantic).
- **[🧠 Semantic Memory Engine](./Module-01-Retrieval-Foundation/04-Vector-Databases/Project/)**
  - *Focus:* Production-grade **ChromaDB** implementation featuring adaptive semantic chunking, multi-format file parsing, and a live Streamlit chat interface.
- **[🕵️ Local Knowledge Explorer (Naive RAG)](./Module-01-Retrieval-Foundation/05-Naive-RAG-Loop/Project/)**
  - *Focus:* End-to-end basic retrieval pipeline with distance thresholds and grounded response logic.

#### 📂 [Module 02: Advanced Retrieval](./Module-02-Advanced-Retrieval/)
*Fixing accuracy issues and optimizing search relevance.*
- **[🔎 OmniSearch: Metadata Filtering](./Module-02-Advanced-Retrieval/01-Metadata-Filtering/Project/)**
  - *Focus:* Self-querying retrieval, global metadata anchors, and defense-in-depth Pydantic validation.
- **Hybrid Search**, **Re-ranking**, and **Query Transformation** (Upcoming).

#### 📂 [Module 03: Structured Logic & Extraction](./Module-03-Structured-Logic-Extraction/)
*Guaranteeing valid outputs and tool integration.*
- **JSON Mode**, **Pydantic/Instructor**, **Function Calling**, and **Guardrails**.

#### 📂 [Module 04: Agentic Workflows](./Module-04-Agentic-Workflows/)
*Autonomous systems that iterate and solve problems.*
- **LangGraph**, **Memory/State Management**, and **Multi-Agent Orchestration**.

#### 📂 [Module 05: Evaluation & LLMOps](./Module-05-Evaluation-LLMOps/)
*Proving reliability with metrics and observability.*
- **RAGAS**, **LLM-as-a-Judge**, and **Observability (LangSmith/Arize)**.

---

## 🛠️ Execution Strategy (Isolated Mastery)
Each module follows a strict folder structure:
- `Learn/`: Deep-dive research, exploratory notebooks, and theory.
- `Project/`: A standalone, functional prototype (CLI or Web App) proving mastery.

---

## 💼 Portfolio & Deployment
- **Interactive Demos:** Hosted on Hugging Face Spaces (Gradio/Streamlit).
- **Local-to-Cloud:** Strategy for moving from **Ollama (Local)** to **Groq/Gemini (Cloud)** seamlessly.
- **Metric-Driven:** Every project includes a **RAGAS score** table to prove performance.

---

## 🚀 Getting Started
1. **Clone the Repo**
2. **Create a Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install Core Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Setup Environment Variables:**
   - Create a `.env` file with your API keys (OpenAI, Gemini, Anthropic, LangSmith, etc.)

---
*Created for a future Senior AI Engineer. Let's build.*
