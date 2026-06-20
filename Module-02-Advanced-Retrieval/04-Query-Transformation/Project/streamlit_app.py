import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

from shared_llm.database import ChromaManager
from shared_llm.processing import DataLoader, Chunker
from shared_llm.retrieval import KeywordEngine
from engine import QueryIntelligenceEngine

# Load environment variables
load_dotenv()

# Configuration
DATA_PATH = "data/"
CHROMA_PATH = "./chroma_db"
BM25_PATH = "./bm25.pkl"
MODEL_NAME = "gpt-oss:120b-cloud" # Default Ollama model

# Ensure directories exist
os.makedirs(DATA_PATH, exist_ok=True)

@st.cache_resource
def get_engine():
    """
    Initializes and caches the Query Intelligence Engine.
    """
    return QueryIntelligenceEngine(
        chroma_path=CHROMA_PATH, 
        bm25_path=BM25_PATH, 
        model=MODEL_NAME
    )

def run_ingestion():
    """
    Handles ingestion of knowledge base into ChromaDB and builds the BM25 index.
    """
    context = DataLoader.master_loader(DATA_PATH, allowed_files=[".json"])
    
    if not context:
        return "No documents found in data directory."

    chroma = ChromaManager(CHROMA_PATH)
    chunker = Chunker(chunk_size=300, chunk_overlap=20)
    
    existing_sources = chroma.get_unique_metadata_values("source")
    new_docs = [doc for doc in context if doc["metadata"]["source"] not in existing_sources]
    
    status_msg = ""
    if new_docs:
        chroma.add_documents(new_docs, chunker=chunker)
        status_msg += f"Successfully ingested new data: {list(set([d['metadata']['source'] for d in new_docs]))}. "
    else:
        status_msg += "No new data to ingest into Chroma. "

    all_data = chroma.collection.get(include=['documents', 'metadatas'])
    formatted_docs = []
    if all_data['ids']:
        for i in range(len(all_data['ids'])):
            formatted_docs.append({
                "content": all_data['documents'][i],
                "metadata": all_data['metadatas'][i]
            })
    
    if formatted_docs:
        kw_engine = KeywordEngine()
        kw_engine.build_index(formatted_docs, save_path=BM25_PATH)
        status_msg += f"BM25 Index synchronized with {len(formatted_docs)} chunks."
    
    return status_msg

# Streamlit UI Setup
st.set_page_config(page_title="Query Intelligence RAG Engine", layout="wide")
st.title("🧠 Query Intelligence RAG Engine")

# Sidebar for Document Ingestion
with st.sidebar:
    st.header("📂 Document Ingestion")
    st.write("Upload `.json` knowledge base files to store them in the data directory and index them.")
    
    uploaded_files = st.file_uploader(
        "Drag and drop files here", 
        type=["json"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("⚡ Process and Index Files"):
            with st.spinner("Saving files and updating indices..."):
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(DATA_PATH, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                # Run ingestion
                result_msg = run_ingestion()
                st.success(result_msg)
                # Clear cached engine to pick up new indices
                st.cache_resource.clear()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input
if user_query := st.chat_input("Ask a question..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_query)
    
    # Format history for the engine
    chat_history = []
    for msg in st.session_state.messages:
        chat_history.append({"role": msg["role"], "content": msg["content"]})
        
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                engine = get_engine()
                response = engine.run(user_query, history=chat_history)
                st.markdown(response)
            except Exception as e:
                response = f"An error occurred: {str(e)}"
                st.error(response)
                
    # Save to session state
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.session_state.messages.append({"role": "assistant", "content": response})
