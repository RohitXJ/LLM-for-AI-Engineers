import time
import os
import streamlit as st
from core import Chroma, Ollama, CrossEnc, CohereReranker, KeywordEngine, RFF, load_json, get_ids, filter_context_by_search

# ── Constants ────────────────────────────────────────────────────────────────
CHROMA_LOC = "./cache/chroma_db"
DATA_DIR   = "./Data"
BM25_CACHE = "./cache/bm25/bm25.pkl"

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentinel Search",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Background */
.stApp { background: #0a0e1a; }

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #0f1729 0%, #1a1f3a 50%, #0f1729 100%);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(99,102,241,0.08) 0%, transparent 60%),
                radial-gradient(circle at 70% 50%, rgba(139,92,246,0.06) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8, #c084fc, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 6px 0;
}
.hero-sub {
    color: #94a3b8;
    font-size: 0.95rem;
    font-weight: 400;
    margin: 0;
}
.badge {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.4);
    color: #818cf8;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 999px;
    margin-right: 6px;
    margin-bottom: 16px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0d1221;
    border-right: 1px solid rgba(99,102,241,0.15);
}
section[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }

/* ── Metric cards ── */
.metric-row { display: flex; gap: 12px; margin-bottom: 24px; }
.metric-card {
    flex: 1;
    background: #111827;
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: rgba(99,102,241,0.5); }
.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #818cf8;
    line-height: 1;
}
.metric-label {
    font-size: 0.72rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 4px;
}

/* ── Pipeline stage cards ── */
.stage-card {
    background: #111827;
    border: 1px solid rgba(30,41,59,0.8);
    border-left: 3px solid;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
    transition: transform 0.15s;
}
.stage-card:hover { transform: translateX(4px); }
.stage-title { font-size: 0.8rem; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }
.stage-time  { font-size: 1.1rem; font-weight: 700; margin: 4px 0 0; }

/* ── Chat bubbles ── */
[data-testid="stChatMessage"] {
    background: #111827 !important;
    border: 1px solid rgba(99,102,241,0.1) !important;
    border-radius: 12px !important;
    margin-bottom: 12px !important;
    padding: 10px !important;
}
[data-testid="stChatMessageContent"] {
    font-size: 0.92rem !important;
    line-height: 1.6 !important;
    color: #e2e8f0 !important;
}
/* User specific style override */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: rgba(79, 70, 229, 0.05) !important;
    border-color: rgba(79, 70, 229, 0.2) !important;
}

.chat-container { display: flex; flex-direction: column; gap: 16px; }
.bubble {
    max-width: 80%;
    padding: 14px 18px;
    border-radius: 14px;
    font-size: 0.92rem;
    line-height: 1.6;
    animation: fadeUp 0.3s ease;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.bubble-user {
    align-self: flex-end;
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: #fff;
    border-bottom-right-radius: 4px;
}
.bubble-ai {
    align-self: flex-start;
    background: #1e293b;
    color: #e2e8f0;
    border: 1px solid rgba(99,102,241,0.2);
    border-bottom-left-radius: 4px;
}
/* Ensure markdown inside bubbles looks good */
.bubble p { margin: 0 0 12px 0; }
.bubble p:last-child { margin-bottom: 0; }
.bubble ul, .bubble ol { margin: 8px 0; padding-left: 24px; }
.bubble li { margin-bottom: 4px; }
.bubble code { 
    background: rgba(0,0,0,0.3); 
    padding: 2px 5px; 
    border-radius: 4px; 
    font-size: 0.85em; 
    font-family: 'JetBrains Mono', 'Fira Code', monospace; 
}
.bubble-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 6px;
    opacity: 0.7;
}

/* ── Section headings ── */
.section-heading {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #475569;
    margin: 20px 0 10px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-heading::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(71,85,105,0.3);
}

/* ── Doc result card ── */
.doc-card {
    background: #111827;
    border: 1px solid rgba(30,41,59,0.9);
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 8px;
    font-size: 0.82rem;
    color: #94a3b8;
    line-height: 1.5;
}
.doc-rank {
    display: inline-block;
    background: rgba(99,102,241,0.2);
    color: #818cf8;
    font-size: 0.68rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 4px;
    margin-bottom: 6px;
}

/* ── Rerank visualizer ── */
.rank-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 12px;
    background: #111827;
    border-radius: 8px;
    margin-bottom: 6px;
    font-size: 0.8rem;
    color: #94a3b8;
}
.rank-badge-before { background: rgba(239,68,68,0.15); color: #f87171; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 0.72rem; }
.rank-badge-after  { background: rgba(34,197,94,0.15); color: #4ade80; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 0.72rem; }
.rank-arrow { color: #475569; }

/* ── Input override ── */
.stTextInput > div > div > input {
    background: #111827 !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    padding: 12px 16px !important;
    font-size: 0.9rem !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextInput > div > div > input:focus {
    border-color: rgba(99,102,241,0.7) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    font-size: 0.88rem !important;
    transition: opacity 0.2s, transform 0.15s !important;
    width: 100% !important;
}
.stButton > button:hover { opacity: 0.9 !important; transform: translateY(-1px) !important; }

/* ── Status tags ── */
.tag {
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 999px;
    margin-right: 4px;
}
.tag-dept  { background: rgba(56,189,248,0.15); color: #38bdf8; border: 1px solid rgba(56,189,248,0.3); }
.tag-cat   { background: rgba(167,139,250,0.15); color: #a78bfa; border: 1px solid rgba(167,139,250,0.3); }
.tag-year  { background: rgba(251,191,36,0.15);  color: #fbbf24; border: 1px solid rgba(251,191,36,0.3); }
.tag-none  { background: rgba(100,116,139,0.15); color: #64748b; border: 1px solid rgba(100,116,139,0.3); }

/* ── Sidebar filter pills ── */
.filter-pill {
    display: inline-block;
    background: rgba(99,102,241,0.1);
    border: 1px solid rgba(99,102,241,0.25);
    color: #818cf8;
    font-size: 0.72rem;
    padding: 3px 10px;
    border-radius: 999px;
    margin: 2px;
}

/* ── System status ── */
.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
}
.dot-green  { background: #4ade80; box-shadow: 0 0 6px #4ade80; }
.dot-yellow { background: #fbbf24; box-shadow: 0 0 6px #fbbf24; }
.dot-red    { background: #f87171; box-shadow: 0 0 6px #f87171; }
</style>
""", unsafe_allow_html=True)


# ── Cached engine init ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def init_engines(model_name: str):
    chroma = Chroma(loc=CHROMA_LOC)
    llm    = Ollama(model=model_name)
    bm25   = KeywordEngine()
    corpus = load_json(DATA_DIR)

    ids         = [item.get('id') for item in corpus]
    fetched_ids = chroma.id_fetch()
    new_ids     = [n for n in ids if n not in fetched_ids]

    if new_ids:
        chroma.data_ingest(new_ids=new_ids, corpus=corpus)
        bm25.from_corpus(corpus, save_path=BM25_CACHE)
    else:
        if not bm25.load(BM25_CACHE):
            bm25.from_corpus(corpus, save_path=BM25_CACHE)

    rff      = RFF(kw_eng=bm25, vec_eng=chroma)
    
    # Initialize both rerankers
    local_reranker = CrossEnc()
    cloud_reranker = CohereReranker()
    
    return chroma, llm, bm25, rff, local_reranker, cloud_reranker, corpus


# ── Session state ─────────────────────────────────────────────────────────────
if "messages"     not in st.session_state: st.session_state.messages     = []
if "ollama_model"  not in st.session_state: st.session_state.ollama_model  = "gpt-oss:20b-cloud"
if "last_metrics" not in st.session_state: st.session_state.last_metrics = None
if "last_filters" not in st.session_state: st.session_state.last_filters = None
if "last_docs"    not in st.session_state: st.session_state.last_docs    = None
if "engine_ready" not in st.session_state: st.session_state.engine_ready = False
if "rerank_mode"   not in st.session_state: st.session_state.rerank_mode   = "Local"


# ── Sidebar Configuration ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; margin-bottom:20px;'>
        <div style='font-size:2rem;'>🛡️</div>
        <div style='font-weight:700; font-size:1rem; color:#818cf8;'>Sentinel Search</div>
        <div style='font-size:0.72rem; color:#475569;'>Hybrid Intelligence Engine</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-heading'>⚙️ Model Configuration</div>", unsafe_allow_html=True)
    st.text_input(
        "Ollama Model Name",
        key="ollama_model",
        help="Enter the exact name of your local Ollama model (e.g. llama3, mistral)"
    )
    
    # Re-ranker Toggle
    st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
    st.markdown("**Cross Encoder Mode**", help="Choose between local processing or cloud-based Cohere Rerank.")
    rerank_mode = st.select_slider(
        "cross_encoder_mode",
        options=["Local", "Cloud"],
        value=st.session_state.rerank_mode,
        label_visibility="collapsed"
    )
    st.session_state.rerank_mode = rerank_mode
    
    # Availability Check
    has_cohere = bool(os.getenv("COHERE_API_KEY"))
    if st.session_state.rerank_mode == "Cloud" and not has_cohere:
        st.warning("⚠️ COHERE_API_KEY not found in .env. Cloud mode will fail.")


# ── Boot engines ──────────────────────────────────────────────────────────────
with st.spinner("🛡️ Initialising Sentinel Search engines…"):
    try:
        chroma, llm, bm25, rff, local_reranker, cloud_reranker, corpus = init_engines(st.session_state.ollama_model)
        st.session_state.engine_ready = True
        filter_meta = chroma.get_filter_values()
    except Exception as e:
        st.error(f"Engine init failed: {e}")
        st.stop()


# ── Sidebar Status & Filters ──────────────────────────────────────────────────
with st.sidebar:
    # System status
    st.markdown("<div class='section-heading'>⚡ System Status</div>", unsafe_allow_html=True)
    doc_count = len(corpus)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{doc_count}</div>
            <div class='metric-label'>Docs Indexed</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        dept_count = len(filter_meta.get("department", []))
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{dept_count}</div>
            <div class='metric-label'>Departments</div>
        </div>""", unsafe_allow_html=True)

    ready_dot = "dot-green" if st.session_state.engine_ready else "dot-red"
    st.markdown(f"""
    <div style='margin-top:12px; font-size:0.8rem; color:#94a3b8;'>
        <span class='status-dot {ready_dot}'></span>Vector DB &nbsp;|&nbsp;
        <span class='status-dot dot-green'></span>BM25 &nbsp;|&nbsp;
        <span class='status-dot dot-green'></span>Cross-Encoder
    </div>
    """, unsafe_allow_html=True)

    # Available metadata
    st.markdown("<div class='section-heading'>🗂️ Available Filters</div>", unsafe_allow_html=True)

    depts = filter_meta.get("department", [])
    cats  = filter_meta.get("category", [])

    if depts:
        st.markdown("**Departments**", help="Auto-detected from corpus")
        pills_html = "".join(f"<span class='filter-pill'>{d}</span>" for d in depts)
        st.markdown(f"<div style='margin-bottom:10px;'>{pills_html}</div>", unsafe_allow_html=True)

    if cats:
        st.markdown("**Categories**")
        pills_html = "".join(f"<span class='filter-pill'>{c}</span>" for c in cats)
        st.markdown(f"<div style='margin-bottom:10px;'>{pills_html}</div>", unsafe_allow_html=True)

    # Live filter applied in last query
    if st.session_state.last_filters:
        st.markdown("<div class='section-heading'>🎯 Last Applied Filter</div>", unsafe_allow_html=True)
        st.json(st.session_state.last_filters)
    else:
        st.markdown("<div class='section-heading'>🎯 Last Applied Filter</div>", unsafe_allow_html=True)
        st.markdown("<span class='tag tag-none'>No filter applied yet</span>", unsafe_allow_html=True)

    # Pipeline legend
    st.markdown("<div class='section-heading'>📐 Pipeline</div>", unsafe_allow_html=True)
    for emoji, label, color in [
        ("1️⃣","Metadata + Vector Search","#818cf8"),
        ("2️⃣","Hybrid RRF Fusion","#38bdf8"),
        ("3️⃣","Cross-Encoder Re-rank","#4ade80"),
    ]:
        st.markdown(f"""
        <div style='font-size:0.78rem; color:#94a3b8; margin-bottom:6px;'>
            {emoji} <span style='color:{color}; font-weight:600;'>{label}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages     = []
        st.session_state.last_metrics = None
        st.session_state.last_filters = None
        st.session_state.last_docs    = None
        st.rerun()


# ── Main layout ───────────────────────────────────────────────────────────────
# Hero
st.markdown("""
<div class='hero'>
    <span class='badge'>Module 02 • Re-ranking</span>
    <span class='badge'>BGE Cross-Encoder</span>
    <span class='badge'>RRF Fusion</span>
    <h1 class='hero-title'>🛡️ Sentinel Search</h1>
    <p class='hero-sub'>Production-grade Hybrid Intelligence Engine — BM25 + Vector + Cross-Encoder Re-ranking over Enterprise Policy Documents</p>
</div>
""", unsafe_allow_html=True)

# Top-level columns: chat | observability
chat_col, obs_col = st.columns([3, 2], gap="large")

# ── CHAT COLUMN ───────────────────────────────────────────────────────────────
with chat_col:
    st.markdown("<div class='section-heading'>💬 Query Interface</div>", unsafe_allow_html=True)

    # Render history
    if st.session_state.messages:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                with st.chat_message("user", avatar="👤"):
                    st.markdown(msg["content"])
            else:
                with st.chat_message("assistant", avatar="🛡️"):
                    st.markdown(msg["content"])
    else:
        st.markdown("""
        <div style='text-align:center; padding:40px 20px; color:#334155;'>
            <div style='font-size:2.5rem; margin-bottom:12px;'>🔍</div>
            <div style='font-size:0.9rem;'>Ask anything about company policies.</div>
            <div style='font-size:0.78rem; margin-top:6px; color:#1e293b;'>
                e.g. "Can a junior dev request a Mac Studio after 6 months?"
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Input
    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
    with st.form("query_form", clear_on_submit=True):
        query = st.text_input(
            "query",
            placeholder="Ask a policy question…",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("⚡ Search & Answer")

    if submitted and query.strip():
        user_msg = query.strip()
        st.session_state.messages.append({"role": "user", "content": user_msg})

        with st.spinner("Running 3-stage pipeline…"):
            try:
                # Phase 1
                t1s = time.perf_counter()
                filters         = llm.generate_filter(user_query=user_msg, existing_metadata=filter_meta)
                context_payload = chroma.query_context(query_text=user_msg, filter_values=filters)
                target_ids      = get_ids(context=context_payload)
                t1e = time.perf_counter()

                if not target_ids:
                    ai_response = "No relevant documents found in the initial search. Try rephrasing your query."
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})
                    st.session_state.last_filters = filters
                    st.rerun()

                # Phase 2
                t2s = time.perf_counter()
                result_ids = rff.search(query=user_msg, target_ids=target_ids, top_k=10)
                candidates = filter_context_by_search(context=context_payload, search_results=result_ids)
                t2e = time.perf_counter()

                # Phase 3
                t3s = time.perf_counter()
                selected_reranker = local_reranker if st.session_state.rerank_mode == "Local" else cloud_reranker
                top_docs = selected_reranker.rank(query=user_msg, candidates=candidates, top_k=2)
                t3e = time.perf_counter()

                # Generation
                final_context = "\n\n".join(top_docs)
                result = llm.chain.invoke({"context": final_context, "question": user_msg})
                ai_response = result.content

                # Store state
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
                st.session_state.last_metrics = {
                    "phase1": t1e - t1s,
                    "phase2": t2e - t2s,
                    "phase3": t3e - t3s,
                    "total":  (t1e - t1s) + (t2e - t2s) + (t3e - t3s),
                }
                st.session_state.last_filters = filters
                # Store candidates with scores for rerank viz
                ranked_list = [(doc_id, score) for doc_id, score in result_ids]
                st.session_state.last_docs = {
                    "candidates": candidates,
                    "rrf_ranked": ranked_list,
                    "top_docs": top_docs,
                }
                st.rerun()

            except Exception as e:
                st.error(f"Pipeline error: {e}")


# ── OBSERVABILITY COLUMN ──────────────────────────────────────────────────────
with obs_col:
    st.markdown("<div class='section-heading'>📊 Live Observability</div>", unsafe_allow_html=True)

    # Performance metrics
    if st.session_state.last_metrics:
        m = st.session_state.last_metrics
        st.markdown(f"""
        <div class='metric-row'>
            <div class='metric-card'>
                <div class='metric-value'>{m['total']:.2f}s</div>
                <div class='metric-label'>Total Latency</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        stages = [
            ("1. Meta + Vector", m['phase1'], "#818cf8"),
            ("2. Hybrid RRF",    m['phase2'], "#38bdf8"),
            ("3. Cross-Encoder", m['phase3'], "#4ade80"),
        ]
        for label, val, color in stages:
            pct = int((val / m['total']) * 100) if m['total'] > 0 else 0
            st.markdown(f"""
            <div class='stage-card' style='border-left-color:{color};'>
                <div class='stage-title'>{label}</div>
                <div class='stage-time' style='color:{color};'>{val:.4f}s
                    <span style='font-size:0.72rem; color:#475569; font-weight:400;'>({pct}%)</span>
                </div>
                <div style='background:#1e293b; border-radius:4px; height:4px; margin-top:8px;'>
                    <div style='background:{color}; width:{pct}%; height:100%; border-radius:4px; opacity:0.7;'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:#111827; border:1px dashed rgba(99,102,241,0.2); border-radius:10px;
                    padding:24px; text-align:center; color:#334155; font-size:0.82rem;'>
            ⏱️ Latency metrics will appear after your first query.
        </div>""", unsafe_allow_html=True)

    # Re-rank visualizer
    st.markdown("<div class='section-heading'>🔀 Re-rank Visualizer</div>", unsafe_allow_html=True)

    if st.session_state.last_docs:
        ld          = st.session_state.last_docs
        rrf_ranked  = ld["rrf_ranked"]
        top_docs    = ld["top_docs"]
        candidates  = ld["candidates"]

        # Map top docs back to ids
        top_doc_ids = set()
        for doc_id, info in candidates.items():
            if info["document"] in top_docs:
                top_doc_ids.add(doc_id)

        for after_rank, (doc_id, rrf_score) in enumerate(rrf_ranked[:8], start=1):
            info     = candidates.get(doc_id, {})
            meta     = info.get("metadata", {})
            dept     = meta.get("department", "—")
            cat      = meta.get("category", "—")
            is_top   = doc_id in top_doc_ids
            glow     = "box-shadow: 0 0 0 1px #4ade80;" if is_top else ""
            top_badge = "<span style='color:#4ade80; font-size:0.68rem; font-weight:700;'>✓ SELECTED</span>" if is_top else ""

            st.markdown(f"""
            <div class='doc-card' style='{glow}'>
                <span class='doc-rank'>#{after_rank}</span> {top_badge}
                <div style='margin-top:4px; font-weight:500; color:#cbd5e1; font-size:0.8rem;'>{doc_id}</div>
                <div style='margin-top:6px;'>
                    <span class='tag tag-dept'>{dept}</span>
                    <span class='tag tag-cat'>{cat}</span>
                </div>
                <div style='margin-top:6px; color:#475569; font-size:0.72rem;'>
                    RRF Score: <span style='color:#818cf8; font-weight:600;'>{rrf_score:.6f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:#111827; border:1px dashed rgba(99,102,241,0.2); border-radius:10px;
                    padding:24px; text-align:center; color:#334155; font-size:0.82rem;'>
            🔀 Re-rank results will appear after your first query.
        </div>""", unsafe_allow_html=True)

    # Top selected docs preview
    if st.session_state.last_docs and st.session_state.last_docs["top_docs"]:
        st.markdown("<div class='section-heading'>📄 Context Sent to LLM</div>", unsafe_allow_html=True)
        for i, doc in enumerate(st.session_state.last_docs["top_docs"], 1):
            preview = doc[:200] + "…" if len(doc) > 200 else doc
            st.markdown(f"""
            <div class='doc-card'>
                <span class='doc-rank'>Context #{i}</span>
                <div style='margin-top:6px; font-size:0.78rem; color:#64748b; line-height:1.5;'>{preview}</div>
            </div>""", unsafe_allow_html=True)
