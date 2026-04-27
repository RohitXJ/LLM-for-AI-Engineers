# Project: The Semantic Memory Engine (Module 1.3 & 1.4)

## 🎯 The Objective
Build a search engine that understands **meaning** rather than just keywords. Instead of matching letters, it matches concepts. This is the foundation of "Long-Term Memory" for any AI agent.

## 🛠️ The "AI Engineering" Stack
- **Embeddings (The Brain):** Use `sentence-transformers` (local) or `OpenAI` (API) to turn text into math vectors.
- **Vector Database (The Library):** Use `ChromaDB` to store and search these vectors.
- **Interface:** A CLI tool for search, with a plan to add a UI (Gradio/Streamlit) using "vibe coding."

---

## 📥 Inputs & 📤 Outputs

### 1. Ingestion Phase (Input)
- **Input:** A folder named `knowledge_base/` containing `.txt` files (e.g., tech notes, recipes, or diary entries).
- **Process:** 
    - Read each file.
    - Break text into chunks (using knowledge from Module 1.2).
    - Convert chunks into **Embeddings** (Module 1.3).
    - Save to **ChromaDB** (Module 1.4).

### 2. Search Phase (Output)
- **User Query:** "How do I make something healthy?"
- **System Action:** Converts the query to a vector and finds the "nearest neighbors" in ChromaDB.
- **Output:** 
    - The top 3 most relevant text snippets.
    - A **Similarity Score** (How confident the AI is).
    - The **Source File Name** where the info was found.

---

## 🖼️ How it should look

### Command Line Interface (CLI)
```bash
> python main.py --query "career advice for engineers"

[Result 1] (Score: 0.89)
"Focus on building a strong foundation in data structures and embeddings..."
Source: notes/mentorship.txt

[Result 2] (Score: 0.75)
"Networking is just as important as coding when looking for senior roles..."
Source: diary/january_goals.txt
```

### Visual Interface (The "Vibe" Goal)
- A clean search bar at the top.
- Cards below showing the results with "Heat Map" colors (Green for high similarity, Yellow for low).

---

## 🚀 Why this makes you "Job Ready"
1. **Infrastructure:** You aren't just calling an API; you are managing a database.
2. **Optimization:** You will learn how to handle "cold starts" (first-time ingestion) vs. "warm searches."
3. **Reasoning:** You'll be able to explain why searching for "King" returns "Queen" even if the words are different.

---

## 📝 Roadmap for Implementation
1. [ ] Setup `ChromaDB` client (Persistent storage).
2. [ ] Create a `collection` for your documents.
3. [ ] Build the `add_document` function (Text -> Embedding -> DB).
4. [ ] Build the `query` function (Query -> Embedding -> DB Search).
5. [ ] (Vibe Coding) Wrap it in a Gradio or Streamlit UI.
