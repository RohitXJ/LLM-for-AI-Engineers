import streamlit as st
import ollama
from core import IngestionManager, HybridRetriever

# -----------------------------------------------------
# CONFIGURATION & CONSTANTS
# -----------------------------------------------------
BM25_PKL = "./SYS_OBJ/BM25/bm25_obj_corpus.pkl"
CHROMA = "./SYS_OBJ/chroma_db"
DATA = "./data"
CHAT_MODEL = "gpt-oss:20b-cloud"

st.set_page_config(page_title="SRE-Pulse: AI RAG Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for a clean, chat-focused layout
st.markdown("""
<style>
    .stChatFloatingInputContainer {
        padding-bottom: 20px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 10px;
        border-left: 5px solid #007bff;
    }
    @media (prefers-color-scheme: dark) {
        .metric-card {
            background-color: #262730;
            border-left: 5px solid #4da3ff;
        }
    }
    .source-tag {
        font-size: 0.8em;
        color: #666;
        background: #eee;
        padding: 2px 6px;
        border-radius: 4px;
        margin-right: 5px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# SYSTEM INITIALIZATION
# -----------------------------------------------------
@st.cache_resource
def get_system():
    manager = IngestionManager(CHROMA, BM25_PKL, DATA)
    manager.sync()
    retriever = HybridRetriever(manager.kw_eng, manager.vec_eng)
    # Cache the actual documents for easy lookup during RAG
    corpus = {doc['id']: doc for doc in manager.load_json_data()}
    return manager, retriever, corpus

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_results" not in st.session_state:
    st.session_state.last_results = {"kw": [], "vec": [], "hybrid": []}

manager, retriever, corpus_map = get_system()

# -----------------------------------------------------
# SIDEBAR: MODEL SELECTION & CONTROLS
# -----------------------------------------------------
with st.sidebar:
    st.title("🤖 SRE-Pulse AI")
    
    # Model Selection (Simplified Manual Override)
    selected_model = st.text_input(
        "Ollama Model", 
        value=CHAT_MODEL,
        help="Type the name of the model you want to use (e.g., llama3.2, mistral)"
    )

    st.divider()
    
    if st.button("🔄 Force Sync Database", use_container_width=True):
        with st.spinner("Syncing..."):
            st.cache_resource.clear()
            st.rerun()

    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_results = {"kw": [], "vec": [], "hybrid": []}
        st.rerun()

    st.divider()
    st.markdown("### 📊 Retrieval Metrics")
    st.caption("Detailed scores from the last query")
    
    # Compact metrics display in sidebar
    if st.session_state.last_results["hybrid"]:
        for doc_id, score in st.session_state.last_results["hybrid"][:3]:
            st.markdown(f"""
            <div class="metric-card">
                <strong>ID: {doc_id}</strong><br>
                RRF Score: <code>{score:.4f}</code>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No retrieval data yet.")

# -----------------------------------------------------
# MAIN UI: CHAT & RAG LOGIC
# -----------------------------------------------------
st.title("🛠️ SRE-Pulse: Intelligent Troubleshooting")
st.caption("Hybrid RAG (BM25 + Vector + RRF) powered by Ollama")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("references"):
            with st.expander("📚 References Found"):
                for ref_id in message["references"]:
                    doc = corpus_map.get(ref_id, {})
                    st.markdown(f"**[{ref_id}] {doc.get('title', 'N/A')}**")
                    st.info(f"**Solution:** {doc.get('solution', 'N/A')}")

# User Input
if prompt := st.chat_input("Describe the SRE issue or error code..."):
    # 1. Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Perform Hybrid Retrieval
    with st.spinner("Searching knowledge base..."):
        kw_results = manager.kw_eng.search(prompt, top_k=5)
        vec_results = manager.vec_eng.search(prompt, top_k=5)
        hybrid_results = retriever.search(prompt, top_k=5)
        
        # Save results for the sidebar metrics
        st.session_state.last_results = {
            "kw": kw_results,
            "vec": vec_results,
            "hybrid": hybrid_results
        }

    # 3. Construct Context for LLM
    context_items = []
    top_ids = [res[0] for res in hybrid_results]
    
    for doc_id in top_ids:
        doc = corpus_map.get(doc_id)
        if doc:
            item = f"ID: {doc_id}\nTitle: {doc['title']}\nError Code: {doc['error_code']}\nContent: {doc['content']}\nSolution: {doc['solution']}"
            context_items.append(item)
    
    context_str = "\n\n---\n\n".join(context_items)

    # 4. Generate Response with Grounding
    with st.chat_message("assistant"):
        if not hybrid_results:
            response = "I'm sorry, I could not find any information related to that query in the SRE database."
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            system_prompt = (
                "You are an SRE Troubleshooting Assistant. Your goal is to provide technical solutions based ONLY on the provided database context.\n"
                "STRICT RULES:\n"
                "1. If the information is not in the context, say: 'I do not have information about this in my database.'\n"
                "2. DO NOT use your own external knowledge.\n"
                "3. Respond in PLAIN TEXT only (no special formatting or markdown tables, but bullet points are okay).\n"
                "4. Be concise and technical.\n"
                "5. Mention the ID of the document you are referencing (e.g., [SRE_001])."
            )
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"DATABASE CONTEXT:\n{context_str}\n\nUSER QUERY: {prompt}"}
            ]
            
            response_placeholder = st.empty()
            full_response = ""
            
            try:
                for chunk in ollama.chat(model=selected_model, messages=messages, stream=True):
                    full_response += chunk['message']['content']
                    response_placeholder.markdown(full_response + "▌")
                
                response_placeholder.markdown(full_response)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response, 
                    "references": top_ids
                })
                # Force refresh to update metrics in sidebar
                st.rerun()
                
            except Exception as e:
                st.error(f"Error calling Ollama: {e}")
