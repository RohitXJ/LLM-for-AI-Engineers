import streamlit as st
import chromadb
import ollama
import os
import datetime
from async_v_main import COLLECTION_NAME, VDB_PATH, CHAT_MODEL, DISTANCE_THRESHOLD, semantic_chunking

# -----------------------------------------------------
# STREAMLIT CONFIGURATION
# -----------------------------------------------------
st.set_page_config(
    page_title="RAG Intelligence Center",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Look
st.markdown("""
<style>
    .source-badge {
        display: inline-block;
        padding: 4px 10px;
        font-size: 0.85em;
        font-weight: 600;
        background-color: #E3F2FD;
        color: #1565C0;
        border-radius: 12px;
        margin-right: 5px;
        border: 1px solid #BBDEFB;
    }
    @media (prefers-color-scheme: dark) {
        .source-badge {
            background-color: #1E3A8A;
            color: #BFDBFE;
            border-color: #1E40AF;
        }
    }
    /* Citation Window Styling */
    .context-box {
        background-color: #f8f9fa;
        border-left: 4px solid #1565C0;
        padding: 15px;
        margin: 5px 0 15px 0;
        border-radius: 6px;
        font-size: 0.9em;
        color: #333;
        overflow-y: visible;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        white-space: pre-wrap;
    }
    @media (prefers-color-scheme: dark) {
        .context-box {
            background-color: #1e1e1e;
            border-left: 4px solid #4da3ff;
            color: #e0e0e0;
        }
    }
    /* Clean headers */
    h1, h2, h3 {
        color: #1E3A8A;
    }
    @media (prefers-color-scheme: dark) {
        h1, h2, h3 {
            color: #BFDBFE;
        }
    }
    
    /* Sticky Context Column */
    div[data-testid="column"]:nth-of-type(2),
    div[data-testid="stColumn"]:nth-of-type(2) {
        position: sticky !important;
        top: 60px !important;
        align-self: flex-start !important;
        max-height: calc(100vh - 100px);
        overflow-y: auto;
        padding-bottom: 20px;
        scrollbar-width: thin;
    }
    
    div[data-testid="column"]:nth-of-type(2)::-webkit-scrollbar,
    div[data-testid="stColumn"]:nth-of-type(2)::-webkit-scrollbar {
        width: 6px;
    }
    div[data-testid="column"]:nth-of-type(2)::-webkit-scrollbar-thumb,
    div[data-testid="stColumn"]:nth-of-type(2)::-webkit-scrollbar-thumb {
        background: rgba(0,0,0,0.2);
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# INITIALIZATION & STATE
# -----------------------------------------------------
@st.cache_resource
def init_vector_db():
    client = chromadb.PersistentClient(path=VDB_PATH)
    return client.get_or_create_collection(name=COLLECTION_NAME)

try:
    collection = init_vector_db()
except Exception as e:
    st.error(f"Failed to initialize Vector Database: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_context" not in st.session_state:
    st.session_state.last_context = ""
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []
if "custom_model" not in st.session_state:
    st.session_state.custom_model = CHAT_MODEL

# -----------------------------------------------------
# CORE RETRIEVAL LOGIC
# -----------------------------------------------------
def retrieve_knowledge(query: str, db_collection: chromadb.Collection, target_sources: list = None):
    where_clause = None
    if target_sources:
        if len(target_sources) == 1:
            where_clause = {"source": target_sources[0]}
        else:
            where_clause = {"source": {"$in": target_sources}}
            
    results = db_collection.query(
        query_texts=[query], 
        n_results=4,
        where=where_clause
    )
    if not results["documents"] or not results["documents"][0]:
        return "", []
        
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]
    
    context_chunks = []
    source_details = []
    seen_sources = set()
    
    for doc, meta, dist in zip(docs, metas, distances):
        if dist <= DISTANCE_THRESHOLD:
            source_name = meta.get('source', 'Unknown File')
            context_chunks.append(f"--- [Source: {source_name}] ---\n{doc}")
            
            if source_name not in seen_sources:
                seen_sources.add(source_name)
                confidence = max(0.0, min(100.0, (1.0 - (dist / 2.0)) * 100))
                source_details.append({
                    "name": source_name,
                    "confidence": confidence
                })
                
    context_str = "\n\n".join(context_chunks)
    return context_str, source_details

# -----------------------------------------------------
# SIDEBAR PANEL (Controls & Ingestion)
# -----------------------------------------------------
with st.sidebar:
    st.title("⚙️ RAG Control Panel")
    
    # 1. Model Configuration
    st.markdown("### 🧠 LLM Selection")
    st.session_state.custom_model = st.text_input(
        "Model Name (Local or Cloud)", 
        value=st.session_state.custom_model,
        help="Type the Ollama model name you want to use (e.g., llama3, mistral, gpt-oss:20b-cloud)."
    )
    
    st.divider()
    
    # 2. Live Document Ingestion (Drag & Drop)
    st.markdown("### 📤 Ingest Documents")
    uploaded_files = st.file_uploader("Upload .txt or .md files", type=["txt", "md"], accept_multiple_files=True)
    if st.button("Process & Embed", use_container_width=True, type="primary") and uploaded_files:
        with st.spinner("Chunking and Embedding..."):
            for f in uploaded_files:
                content = f.read().decode("utf-8")
                filename = f.name
                # Check if exists
                existing = collection.get(where={"source": filename}, limit=1)
                if existing["ids"]:
                    st.warning(f"File '{filename}' is already in the database.")
                    continue
                
                # Chunk and Add
                chunks = semantic_chunking(content)
                ids, docs, metas = [], [], []
                ext = os.path.splitext(filename)[1]
                mod_date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                for i, chunk in enumerate(chunks):
                    ids.append(f"{filename}_chunk_{i+1}")
                    docs.append(chunk.page_content)
                    metas.append({"source": filename, "extension": ext, "last_modified": mod_date, "chunk_index": i+1})
                if ids:
                    collection.upsert(ids=ids, documents=docs, metadatas=metas)
                st.success(f"Ingested '{filename}' ({len(chunks)} chunks).")
        
    st.divider()
    
    # 3. Document Management (Delete)
    st.markdown("### 🗑️ Manage Database")
    all_docs = collection.get(include=["metadatas"])
    if all_docs and all_docs.get("metadatas"):
        unique_sources = sorted(list(set(m["source"] for m in all_docs["metadatas"] if m and "source" in m)))
        to_delete = st.multiselect("Select files to remove", unique_sources)
        if st.button("Delete Selected Documents", use_container_width=True) and to_delete:
            with st.spinner("Removing from Vector DB..."):
                for source in to_delete:
                    collection.delete(where={"source": source})
            st.success("Documents deleted.")
            st.rerun()
    else:
        st.info("Vector database is currently empty.")
        
    st.divider()
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_context = ""
        st.session_state.last_sources = []
        st.rerun()

# -----------------------------------------------------
# MAIN INTERFACE (Chat & Context Viewer)
# -----------------------------------------------------
st.title("🏛️ Intelligent Document Assistant")
st.markdown("Ask questions, and the AI will analyze your ingested documents to provide grounded answers.")
st.write("") # Spacer

# Split into two columns: 65% Chat, 35% Live Context
chat_col, context_col = st.columns([65, 35])

# ---- RIGHT COLUMN: LIVE CONTEXT VIEWER ----
with context_col:
    st.subheader("🔍 Live Context View")
    st.caption("Raw retrieval data the AI is using for the latest query.")
    if st.session_state.last_context:
        st.markdown(f'<div class="context-box">{st.session_state.last_context}</div>', unsafe_allow_html=True)
        st.markdown("**Matched Files:**")
        for src in st.session_state.last_sources:
            st.markdown(
                f"<span class='source-badge'>{src['name']}</span> (`{src['confidence']:.1f}%` match)", 
                unsafe_allow_html=True
            )
    else:
        st.info("No context retrieved yet. Ask a question to view the Vector Database output.")

# ---- LEFT COLUMN: CHAT INTERFACE ----
with chat_col:
    # Display available tags so the user knows what they can type
    all_docs_meta = collection.get(include=["metadatas"])
    if all_docs_meta and all_docs_meta.get("metadatas"):
        unique_sources_hint = sorted(list(set(m["source"] for m in all_docs_meta["metadatas"] if m and "source" in m)))
        if unique_sources_hint:
            st.caption(f"🏷️ **Available tags:** {', '.join([f'`@{s}`' for s in unique_sources_hint])}")
    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📑 Sources Used"):
                    for src in msg["sources"]:
                        st.markdown(f"- **{src['name']}** (Relevance: `{src['confidence']:.1f}%`)")

    # User Input
    if prompt := st.chat_input("Ask about your documents..."):
        # Render user prompt
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        # Detect tags directly from the user's query
        all_docs_meta = collection.get(include=["metadatas"])
        target_sources = []
        
        if all_docs_meta and all_docs_meta.get("metadatas"):
            unique_sources = set(m["source"] for m in all_docs_meta["metadatas"] if m and "source" in m)
            for src in unique_sources:
                # Support @filename.txt or just filename.txt anywhere in the prompt
                if f"{src}" in prompt or src in prompt:
                    target_sources.append(src)
                    
        # Determine the query to send to the Vector DB
        search_query_stripped = prompt
        for src in target_sources:
            search_query_stripped = search_query_stripped.replace(f"@{src}", "").replace(src, "").strip()
            
        # If the user ONLY typed the tag without a real question, use a fallback summary query
        # Otherwise, keep the original prompt completely intact so the Vector DB gets all semantic keywords!
        vector_query = prompt if search_query_stripped else "What is the summary and key information of this document?"

        # Generate assistant response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Searching vector space..."):
                context, sources = retrieve_knowledge(vector_query, collection, target_sources=target_sources)
                
            # Update session state for the Context Viewer
            st.session_state.last_context = context
            st.session_state.last_sources = sources
            
            if not context:
                no_data_msg = "I could not find any relevant information in the knowledge base that passes the retrieval threshold to answer your query."
                st.warning(no_data_msg)
                st.session_state.messages.append({"role": "assistant", "content": no_data_msg, "sources": []})
                st.rerun() # Refresh the right column immediately
            else:
                system_prompt = (
                    "You are an expert Document Intelligence Assistant. Answer the user's question "
                    "ONLY using the provided context. If the information is not present, clearly state "
                    "that you don't have enough information. Format your answer cleanly. "
                    "Always try to cite the sources you used at the end of the relevant sentences in brackets like [Source: filename.txt]."
                )
                
                messages_payload = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {prompt}"}
                ]
                
                response_placeholder = st.empty()
                full_response = ""
                
                try:
                    # Stream the response from the customized Ollama model
                    for chunk in ollama.chat(model=st.session_state.custom_model, messages=messages_payload, stream=True):
                        delta = chunk['message']['content']
                        full_response += delta
                        response_placeholder.markdown(full_response + "▌")
                    
                    response_placeholder.markdown(full_response)
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": full_response, 
                        "sources": sources
                    })
                    st.rerun() # Refresh the UI to update the right Context column
                    
                except Exception as e:
                    st.error(f"Error communicating with LLM '{st.session_state.custom_model}': {str(e)}")
