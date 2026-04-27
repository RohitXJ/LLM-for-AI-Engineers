import streamlit as st
import os
import asyncio
import tempfile
from pathlib import Path
from helper import knowledge_chk, extract_text, dump_to_vector_DB, chroma_init

# --- Page Configuration ---
st.set_page_config(
    page_title="Semantic Memory Engine",
    page_icon="🧠",
    layout="wide"
)

# --- Styling ---
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- Initialization ---
if "collection" not in st.session_state:
    st.session_state.collection = chroma_init()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# --- App Header ---
st.markdown('<div class="main-header">🧠 Semantic Memory Engine</div>', unsafe_allow_html=True)
st.markdown("Upload your documents (PDF, TXT, DOCX) and chat with your knowledge base.")

# --- Sidebar / Controls ---
with st.sidebar:
    st.header("📂 Data Control Center")
    uploaded_files = st.file_uploader(
        "Upload Documents", 
        type=["pdf", "txt", "docx", "md"], 
        accept_multiple_files=True
    )
    
    process_btn = st.button("🚀 Process & Index Files", use_container_width=True, type="primary")
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- File Processing Logic ---
async def process_files(files):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Use a temp directory to store files for extraction
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, uploaded_file in enumerate(files):
            file_name = uploaded_file.name
            
            if file_name in st.session_state.processed_files:
                continue
                
            status_text.text(f"Processing: {file_name}...")
            
            # Save to temporary path
            tmp_path = os.path.join(tmp_dir, file_name)
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                # Extract and Index
                documents = await extract_text(tmp_path)
                if documents:
                    await dump_to_vector_DB(st.session_state.collection, documents)
                    st.session_state.processed_files.add(file_name)
                else:
                    st.warning(f"Skipped {file_name}: No extractable text.")
            except Exception as e:
                st.error(f"Error processing {file_name}: {e}")
            
            progress_bar.progress((i + 1) / len(files))
            
    status_text.text("✅ All files indexed successfully!")
    st.success("Indexing complete! You can now query your documents.")

if process_btn and uploaded_files:
    asyncio.run(process_files(uploaded_files))
elif process_btn and not uploaded_files:
    st.warning("Please upload some files first.")

# --- Chat Interface ---
st.divider()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "results" in message:
            with st.expander("View Sources"):
                for i, res in enumerate(message["results"]):
                    st.write(f"**Source {i+1}:** {res['source']} (Score: {res['score']:.4f})")
                    st.info(res['content'])

# React to user input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Query ChromaDB
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            results = st.session_state.collection.query(
                query_texts=[prompt],
                n_results=3
            )
            
            if results['ids'] and results['ids'][0]:
                response = f"I found some relevant information in your documents:"
                st.markdown(response)
                
                # Format sources for display
                source_data = []
                for i in range(len(results['ids'][0])):
                    source_data.append({
                        "source": results['metadatas'][0][i].get('source'),
                        "score": results['distances'][0][i],
                        "content": results['documents'][0][i]
                    })
                
                with st.expander("View Sources"):
                    for i, res in enumerate(source_data):
                        st.write(f"**Source {i+1}:** {res['source']} (Score: {res['score']:.4f})")
                        st.info(res['content'])
                
                # Save assistant response to history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "results": source_data
                })
            else:
                error_msg = "I couldn't find any relevant information in the uploaded documents."
                st.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
