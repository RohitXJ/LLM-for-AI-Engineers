"""
OmniSearch RAG Dashboard
========================
A professional Streamlit interface for the Metadata-Filtering RAG system.
All processing logic is preserved exactly from app.py / core/*.
"""

MARKDOWN_HINT = (
    "\n\n[Format instruction: Use Markdown in your response where appropriate — "
    "headers, bullet points, bold, code blocks — especially for complex topics.]"
)

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path

import streamlit as st

# ── Make sure `core` is importable when running from any cwd ──────────────────
sys.path.insert(0, str(Path(__file__).parent))
from core import read_data, Chunker, Chroma_VDB, Ollama

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS  (same as app.py)
# ─────────────────────────────────────────────────────────────────────────────
COLLECTION_NAME = "SMART_DOC_CHAT"
VDB_PATH        = str(Path(__file__).parent / "chroma_db")
METADATA_FIELDS = ["topic", "year", "complexity", "priority", "audience"]

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OmniSearch · RAG Dashboard",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Root palette ── */
:root {
    --bg-base:      #0f1117;
    --bg-surface:   #1a1d27;
    --bg-card:      #21253a;
    --accent:       #4f8ef7;
    --accent-light: #7aaeff;
    --accent-glow:  rgba(79,142,247,.18);
    --success:      #34d399;
    --warn:         #fbbf24;
    --danger:       #f87171;
    --text-primary: #e8eaf0;
    --text-muted:   #8892a4;
    --border:       rgba(255,255,255,.07);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* ── Main area ── */
.main .block-container { padding: 1.4rem 2rem 2rem; max-width: 100%; }

/* ── Brand header ── */
.brand-header {
    display: flex; align-items: center; gap: .75rem;
    padding: .5rem 0 1.2rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.4rem;
}
.brand-icon { font-size: 2rem; }
.brand-title { font-size: 1.55rem; font-weight: 700; color: var(--text-primary); letter-spacing: -.02em; }
.brand-subtitle { font-size: .8rem; color: var(--text-muted); margin-top: .1rem; }
.brand-badge {
    margin-left: auto;
    background: var(--accent-glow);
    border: 1px solid var(--accent);
    color: var(--accent-light);
    font-size: .7rem; font-weight: 600;
    padding: .25rem .7rem; border-radius: 20px; letter-spacing: .05em;
}

/* ── Section labels ── */
.section-label {
    font-size: .7rem; font-weight: 600; letter-spacing: .1em;
    color: var(--text-muted); text-transform: uppercase;
    margin: 1.2rem 0 .5rem;
}

/* ── Cards ── */
.info-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.1rem;
    margin-bottom: .8rem;
}

/* ── Filter pills ── */
.filter-grid { display: flex; flex-wrap: wrap; gap: .4rem; margin-top: .4rem; }
.pill {
    font-size: .72rem; font-weight: 500; padding: .22rem .65rem;
    border-radius: 20px; border: 1px solid;
    white-space: nowrap;
}
.pill-topic     { background: rgba(79,142,247,.15); border-color: #4f8ef7; color: #7aaeff; }
.pill-year      { background: rgba(52,211,153,.15); border-color: #34d399; color: #6ee7b7; }
.pill-complexity{ background: rgba(251,191,36,.15);  border-color: #fbbf24; color: #fde68a; }
.pill-priority  { background: rgba(248,113,113,.15); border-color: #f87171; color: #fca5a5; }
.pill-audience  { background: rgba(167,139,250,.15); border-color: #a78bfa; color: #c4b5fd; }

/* ── Stat boxes ── */
.stat-row { display: flex; gap: .6rem; margin-bottom: 1rem; }
.stat-box {
    flex: 1; background: var(--bg-card);
    border: 1px solid var(--border); border-radius: 8px;
    padding: .6rem .8rem; text-align: center;
}
.stat-value { font-size: 1.3rem; font-weight: 700; color: var(--accent-light); }
.stat-label { font-size: .65rem; color: var(--text-muted); margin-top: .1rem; }

/* ── Native chat message overrides ── */
[data-testid="stChatMessage"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    margin-bottom: .5rem !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: rgba(79,142,247,.1) !important;
    border-color: rgba(79,142,247,.3) !important;
}
.chat-meta {
    font-size: .68rem; color: var(--text-muted);
    margin-top: .3rem; padding: .25rem .55rem;
    background: rgba(0,0,0,.3); border-radius: 6px;
    font-family: monospace;
}
.chat-empty {
    text-align: center; padding: 3rem 1rem;
    color: var(--text-muted); font-size: .9rem;
}
.md-hint {
    display: flex; align-items: center; gap: .45rem;
    font-size: .72rem; color: var(--accent-light);
    background: var(--accent-glow);
    border: 1px solid rgba(79,142,247,.25);
    border-radius: 6px; padding: .3rem .7rem;
    margin-bottom: .75rem;
}
.danger-btn > button {
    background: rgba(248,113,113,.15) !important;
    color: #f87171 !important;
    border: 1px solid rgba(248,113,113,.35) !important;
}
.danger-btn > button:hover { background: rgba(248,113,113,.3) !important; }

/* ── Upload drop zone ── */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(79,142,247,.4) !important;
    border-radius: 10px !important;
    background: rgba(79,142,247,.04) !important;
    transition: border-color .2s;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent) !important; }

/* ── Streamlit widget overrides ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div { background: var(--bg-card) !important; border-color: var(--border) !important; color: var(--text-primary) !important; }
.stButton > button {
    background: var(--accent) !important; color: #fff !important;
    border: none !important; font-weight: 600 !important;
    border-radius: 8px !important; transition: opacity .15s;
}
.stButton > button:hover { opacity: .85 !important; }
.stSuccess { background: rgba(52,211,153,.1) !important; border-left: 3px solid var(--success) !important; }
.stError   { background: rgba(248,113,113,.1) !important; border-left: 3px solid var(--danger) !important; }
.stWarning { background: rgba(251,191,36,.1)  !important; border-left: 3px solid var(--warn) !important; }
div[data-testid="stExpander"] { background: var(--bg-card) !important; border: 1px solid var(--border) !important; border-radius: 8px !important; }

/* ── Scrollable chat container ── */
.chat-scroll {
    max-height: 460px; overflow-y: auto;
    padding-right: .3rem;
    scrollbar-width: thin;
    scrollbar-color: var(--accent) var(--bg-surface);
}

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1rem 0 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "chat_history":   [],   # list of dicts {role, content, filters, latency}
        "llm":            None,
        "chroma":         None,
        "chunker":        None,
        "model_name":     "gpt-oss:20b-cloud",
        "ingest_log":     [],
        "db_stats":       {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ─────────────────────────────────────────────────────────────────────────────
# CORE COMPONENT FACTORY  (cached per model name)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_components(model_name: str):
    llm     = Ollama(model=model_name)
    chroma  = Chroma_VDB(DB_PATH=VDB_PATH, collection_name=COLLECTION_NAME)
    chunker = Chunker()
    return llm, chroma, chunker

# ─────────────────────────────────────────────────────────────────────────────
# ASYNC HELPERS  (run in thread-pool via asyncio to keep UI responsive)
# ─────────────────────────────────────────────────────────────────────────────
async def async_extract_doc_metadata(llm: Ollama, content: str) -> dict:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, llm.extract_document_metadata, content)

async def async_extract_chunk_meta(llm: Ollama, chunk: str, global_meta: dict) -> dict:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, llm.extract_metadata, chunk, global_meta)

async def async_ingest_file(llm: Ollama, chroma: Chroma_VDB, chunker: Chunker,
                            file_path: str, orig_name: str) -> str:
    """
    Mirrors data_ingest() from app.py — logic unchanged.
    Returns a status message string.
    """
    raw_data  = read_data(file_path=file_path)
    raw_data["metadata"]["source"] = orig_name
    file_name = orig_name

    if chroma.file_check_query(file_name):
        return f"⚠️ **{file_name}** already indexed — skipped."

    # Step 1: Global anchor metadata
    global_meta = await async_extract_doc_metadata(llm, raw_data["content"])

    # Step 2: Chunk
    chunks, ids = chunker.chunk(raw_text=raw_data["content"])
    file_meta   = {**global_meta, **raw_data["metadata"]}

    # Step 3: Per-chunk metadata (concurrent)
    chunk_meta_tasks = [
        async_extract_chunk_meta(llm, chunk, file_meta) for chunk in chunks
    ]
    chunk_metas = await asyncio.gather(*chunk_meta_tasks)

    # Step 4: Build final ids & metadata lists (same as data_ingest in app.py)
    final_ids  = []
    final_meta = []
    for chunk, c_id, meta_ext in zip(chunks, ids, chunk_metas):
        final_ids.append(f"{file_meta['source']}_{c_id}")
        final_meta.append({**meta_ext, **file_meta})

    # Step 5: Ingest
    try:
        chroma.context_ingest(ids=final_ids, docs=chunks, meta=final_meta)
        return f"✅ **{file_name}** — {len(chunks)} chunks indexed."
    except Exception as e:
        return f"❌ **{file_name}** — ingest failed: {e}"


async def async_query(llm: Ollama, chroma: Chroma_VDB, query: str):
    """
    Mirrors the query loop in app.py — logic unchanged.
    Returns (answer_str, filters_dict, context_str).
    """
    loop = asyncio.get_event_loop()

    # Step 3 (app.py): Fetch existing tags
    existing_topics = await loop.run_in_executor(None, chroma.get_unique_values, "topic")
    existing_years  = await loop.run_in_executor(None, chroma.get_unique_values, "year")

    # Step 4: Self-querying filter
    filters = await loop.run_in_executor(
        None, llm.generate_filter, query,
        {"topic": existing_topics, "year": existing_years}
    )

    # Query vector DB
    results = await loop.run_in_executor(
        None, chroma.context_query, query, filters
    )

    if not results["documents"][0]:
        return (
            "I'm sorry, no relevant information was found in the database with those filters.",
            filters,
            ""
        )

    context = "\n\n".join(results["documents"][0])

    # Append markdown formatting hint at dashboard level (core logic unchanged)
    enriched_question = query + MARKDOWN_HINT
    result = await loop.run_in_executor(
        None, llm.chain.invoke, {"context": context, "question": enriched_question}
    )
    return result.content, filters, context

# ─────────────────────────────────────────────────────────────────────────────
# DB FILE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def get_indexed_files(chroma: Chroma_VDB) -> list:
    """Returns sorted list of unique source file names indexed in ChromaDB."""
    try:
        results = chroma.collection.get(include=["metadatas"])
        sources = {m.get("source") for m in results["metadatas"] if m and m.get("source")}
        return sorted(list(sources))
    except Exception:
        return []

def delete_file_from_db(chroma: Chroma_VDB, source_name: str) -> bool:
    """Deletes all chunks for a given source file from ChromaDB."""
    try:
        chroma.collection.delete(where={"source": source_name})
        return True
    except Exception as e:
        st.error(f"Delete failed: {e}")
        return False

# ─────────────────────────────────────────────────────────────────────────────
# METADATA PANEL HELPER
# ─────────────────────────────────────────────────────────────────────────────
def get_db_stats(chroma: Chroma_VDB) -> dict:
    stats = {}
    for field in METADATA_FIELDS:
        stats[field] = chroma.get_unique_values(field)
    try:
        stats["_total_chunks"] = len(chroma.collection.get()["ids"])
    except Exception:
        stats["_total_chunks"] = 0
    return stats

PILL_CLASS = {
    "topic":      "pill-topic",
    "year":       "pill-year",
    "complexity": "pill-complexity",
    "priority":   "pill-priority",
    "audience":   "pill-audience",
}

def render_filter_panel(chroma: Chroma_VDB):
    stats = get_db_stats(chroma)
    total = stats.pop("_total_chunks", 0)

    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-box">
            <div class="stat-value">{total}</div>
            <div class="stat-label">Total Chunks</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{len(stats.get('topic', []))}</div>
            <div class="stat-label">Topics</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{len(stats.get('year', []))}</div>
            <div class="stat-label">Years</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    for field, values in stats.items():
        pill_cls = PILL_CLASS.get(field, "pill-topic")
        label    = field.capitalize()
        if not values:
            continue
        pills_html = "".join(
            f'<span class="pill {pill_cls}">{v}</span>' for v in values
        )
        st.markdown(f"""
        <div class="info-card">
            <div class="section-label">{label}</div>
            <div class="filter-grid">{pills_html}</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ── SIDEBAR PART 1 — Model Configuration (no chroma needed)
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-label">⚙️ Model Configuration</div>', unsafe_allow_html=True)

    model_input = st.text_input(
        "Ollama Model Name",
        value=st.session_state["model_name"],
        placeholder="e.g. llama3, mistral, gpt-oss:20b-cloud",
        help="Enter any model name served by your local Ollama instance.",
        key="model_input_widget"
    )
    if st.button("Apply Model", use_container_width=True):
        st.session_state["model_name"] = model_input.strip() or "gpt-oss:20b-cloud"
        load_components.clear()
        st.success(f"Model set to **{st.session_state['model_name']}**")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────
try:
    llm, chroma, chunker = load_components(st.session_state["model_name"])
except Exception as e:
    st.error(f"Failed to load components: {e}")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# ── SIDEBAR PART 2 — DB Control Panel + Clear Chat (chroma now available)
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("---")
    st.markdown('<div class="section-label">🗄️ Database Control Panel</div>', unsafe_allow_html=True)

    indexed_files = get_indexed_files(chroma)
    if indexed_files:
        selected_file = st.selectbox(
            "Select a document",
            options=indexed_files,
            label_visibility="collapsed",
            key="db_file_select"
        )
        # Show chunk count for selected file
        try:
            count = len(chroma.collection.get(where={"source": selected_file})["ids"])
            st.markdown(
                f'<div style="font-size:.72rem; color:var(--text-muted); margin:.3rem 0 .6rem;">'
                f'📄 {selected_file} &nbsp;·&nbsp; {count} chunks</div>',
                unsafe_allow_html=True
            )
        except Exception:
            pass
        st.markdown('<div class="danger-btn">', unsafe_allow_html=True)
        if st.button("🗑️ Delete This Document", use_container_width=True, key="delete_doc_btn"):
            if delete_file_from_db(chroma, selected_file):
                st.success(f"Deleted **{selected_file}**")
                load_components.clear()
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="font-size:.8rem; color:var(--text-muted); padding:.4rem 0;">'  
            'No documents indexed yet.</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state["chat_history"] = []
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# BRAND HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="brand-header">
    <div class="brand-icon">🔎</div>
    <div>
        <div class="brand-title">OmniSearch RAG Dashboard</div>
        <div class="brand-subtitle">Metadata-Filtered Document Intelligence · Module 02</div>
    </div>
    <div class="brand-badge">ADVANCED RETRIEVAL</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TWO-COLUMN LAYOUT:  left (ingestion + chat)  |  right (filter panel)
# ─────────────────────────────────────────────────────────────────────────────
col_main, col_filters = st.columns([2.6, 1], gap="large")

# ══════════════════════════════════════════════════
#  RIGHT COLUMN — Live Filter Panel
# ══════════════════════════════════════════════════
with col_filters:
    st.markdown('<div class="section-label">🏷️ Live Database Filters</div>', unsafe_allow_html=True)

    refresh_filters = st.button("↻ Refresh", use_container_width=True, key="refresh_filters")
    if refresh_filters or "filter_stats_cache" not in st.session_state:
        st.session_state["filter_stats_cache"] = True  # trigger render

    render_filter_panel(chroma)

# ══════════════════════════════════════════════════
#  LEFT COLUMN — Ingestion + Chat
# ══════════════════════════════════════════════════
with col_main:

    # ── SECTION 1: Document Ingestion ─────────────────
    st.markdown('<div class="section-label">📂 Document Ingestion</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Drag & drop `.txt` or `.md` files — they will be chunked, tagged, and indexed automatically.",
        type=["txt", "md"],
        accept_multiple_files=True,
        label_visibility="visible",
        key="file_uploader"
    )

    if uploaded_files:
        if st.button(f"⚡ Process {len(uploaded_files)} File(s)", use_container_width=False):
            progress_bar = st.progress(0, text="Starting ingestion…")
            log_area     = st.empty()
            log_lines    = []

            async def run_ingestion():
                tasks = []
                tmp_paths = []
                for uf in uploaded_files:
                    suffix = Path(uf.name).suffix
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=suffix,
                        prefix=Path(uf.name).stem + "_"
                    ) as tmp:
                        tmp.write(uf.read())
                        tmp_paths.append((uf.name, tmp.name))

                results = []
                for idx, (orig_name, tmp_path) in enumerate(tmp_paths):
                    msg = await async_ingest_file(llm, chroma, chunker, tmp_path, orig_name)
                    results.append(msg)
                    pct = int((idx + 1) / len(tmp_paths) * 100)
                    progress_bar.progress(pct, text=f"Processing {orig_name}…")
                    os.unlink(tmp_path)
                return results

            msgs = asyncio.run(run_ingestion())
            progress_bar.empty()

            for m in msgs:
                st.markdown(m)
            st.session_state["ingest_log"].extend(msgs)

    # Ingestion log expander
    if st.session_state["ingest_log"]:
        with st.expander("📋 Ingestion Log", expanded=False):
            for entry in reversed(st.session_state["ingest_log"][-20:]):
                st.markdown(f"<div style='font-size:.82rem; padding:.2rem 0;'>{entry}</div>",
                            unsafe_allow_html=True)

    st.markdown("---")

    # ── SECTION 2: Chat Interface ──────────────────────
    st.markdown('<div class="section-label">💬 Document Chat</div>', unsafe_allow_html=True)

    # Markdown support hint banner
    st.markdown(
        '<div class="md-hint">✦ Markdown replies are supported &amp; enabled — '
        'the AI will use headers, lists, and code blocks for complex topics.</div>',
        unsafe_allow_html=True
    )

    # Chat history display using st.chat_message for native markdown rendering
    if not st.session_state["chat_history"]:
        st.markdown("""
        <div class="chat-empty">
            🤖 Ask anything about your indexed documents.<br>
            <span style="font-size:.8rem;">Metadata filters are applied automatically to every query.</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state["chat_history"]:
            if msg["role"] == "user":
                with st.chat_message("user", avatar="🧑"):
                    st.markdown(msg["content"])
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    # st.markdown renders full markdown: headers, bold, code, tables
                    st.markdown(msg["content"])
                    filter_str = str(msg.get("filters", "None"))
                    latency    = msg.get("latency", "")
                    st.markdown(
                        f'<div class="chat-meta">🔍 Filters: {filter_str} &nbsp;·&nbsp; ⏱ {latency}</div>',
                        unsafe_allow_html=True
                    )

    # Query input form
    st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)
    with st.form(key="chat_form", clear_on_submit=True):
        query_col, btn_col = st.columns([5, 1])
        with query_col:
            user_query = st.text_input(
                "Your question",
                placeholder="e.g. What are the key cybersecurity risks from 2023?",
                label_visibility="collapsed",
                key="query_input"
            )
        with btn_col:
            submit = st.form_submit_button("Ask →", use_container_width=True)

    if submit and user_query.strip():
        st.session_state["chat_history"].append({"role": "user", "content": user_query.strip()})

        with st.spinner("Searching and generating answer…"):
            t0 = time.perf_counter()
            answer, filters, _ = asyncio.run(async_query(llm, chroma, user_query.strip()))
            latency = f"{time.perf_counter() - t0:.1f}s"

        st.session_state["chat_history"].append({
            "role": "assistant", "content": answer,
            "filters": filters, "latency": latency,
        })
        st.rerun()
