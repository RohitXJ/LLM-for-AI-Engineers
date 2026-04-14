import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import models

# ----------------- Configuration & Data -----------------

st.set_page_config(
    page_title="LLM Budgeting Tool",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODELS = {
    "GPT-4o": {"tier": "Frontier", "code_name": "o200k_base", "cost_per_1m": 5.00, "context_limit": 128000},
    "o1": {"tier": "Frontier", "code_name": "o200k_base", "cost_per_1m": 15.00, "context_limit": 200000},
    "GPT-4 Turbo": {"tier": "Frontier", "code_name": "cl100k_base", "cost_per_1m": 10.00, "context_limit": 128000},
    "GPT-4o-mini": {"tier": "Budget", "code_name": "o200k_base", "cost_per_1m": 0.15, "context_limit": 128000},
    "Llama-3.1": {"tier": "Budget", "code_name": "unsloth/Meta-Llama-3.1-8B", "cost_per_1m": 0.10, "context_limit": 128000},
    "Llama-3.3": {"tier": "Budget", "code_name": "unsloth/Meta-Llama-3.1-8B", "cost_per_1m": 0.50, "context_limit": 128000},
    "DeepSeek-V3": {"tier": "Budget", "code_name": "deepseek-ai/DeepSeek-V3", "cost_per_1m": 0.14, "context_limit": 128000},
    "Legacy-Davinci": {"tier": "Legacy", "code_name": "p50k_base", "cost_per_1m": 20.00, "context_limit": 4096},
}

# ----------------- UI Styling -----------------
st.markdown("""
    <style>
        .stProgress > div > div > div > div {
            background-color: #4CAF50;
        }
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            color: #E2E8F0;
            margin-bottom: 0rem;
        }
        .sub-header {
            font-size: 1.2rem;
            color: #94A3B8;
            margin-bottom: 2rem;
        }
    </style>
""", unsafe_allow_html=True)


# ----------------- Helper Functions -----------------

@st.cache_resource(show_spinner=False)
def load_encoder(model_code_name):
    """Caches the encoder so it doesn't reload heavily every time."""
    return models.models_call(model_code_name)

def get_token_count(text, model_code_name):
    if not text.strip():
        return 0
    encoder = load_encoder(model_code_name)
    if not encoder:
        st.error(f"Failed to load encoder `{model_code_name}`. It might be gated or unavailable.")
        return 0
    if hasattr(encoder, "encode"):
        try:
            return len(encoder.encode(text))
        except Exception as e:
            st.error(f"Encoding Error ({model_code_name}): {e}")
            return 0
    else:
        try:
           return len(encoder(text)["input_ids"])
        except Exception as e:
           st.error(f"Encoding Error ({model_code_name}): {e}")
           return 0


# ----------------- Application Logic -----------------

st.markdown('<p class="main-header">💸 The LLM Budgeting Tool</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Automated Token Auditing & Cost Projection Dashboard for Senior AI Engineers.</p>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("Select one or multiple models to compare tokenizer efficiency and costs.")
    selected_models = st.multiselect(
        "Compare Models:", 
        options=list(MODELS.keys()), 
        default=["GPT-4o", "GPT-4o-mini", "Llama-3.1"]
    )
    st.divider()
    st.markdown("### Quick Info")
    st.info("""
    **Frontier Models:** High reasoning, higher cost.
    **Budget Models:** Great for routine extraction & RAG.
    """)

st.write("### 1. Paste Your Text Payload")
text_input = st.text_area("Input your raw logs, prompt instructions, or context documents below:", height=250, placeholder="Paste your text here...")

if text_input and len(selected_models) > 0:
    st.divider()
    
    # Process text
    word_count = len(text_input.split())
    
    # Gather Data
    results = []
    
    with st.spinner("Analyzing tokens across selected models..."):
        for model_name in selected_models:
            model_info = MODELS[model_name]
            token_count = get_token_count(text_input, model_info["code_name"])
            density = token_count / word_count if word_count > 0 else 0
            
            fill_ratio = token_count / model_info["context_limit"]
            
            # Costs
            cost_1m = model_info["cost_per_1m"]
            cost_actual = (token_count / 1_000_000) * cost_1m
            cost_10m = 10 * cost_1m
            cost_100m = 100 * cost_1m
            
            results.append({
                "Model": model_name,
                "Tier": model_info["tier"],
                "Tokens": token_count,
                "Density": round(density, 2),
                "Context Fill %": min(fill_ratio, 1.0),
                "Context Limit": model_info["context_limit"],
                "Cost per 1M ($)": cost_1m,
                "Actual Payload ($)": cost_actual,
                "Proj. 10M ($)": cost_10m,
                "Proj. 100M ($)": cost_100m,
            })
    
    df = pd.DataFrame(results)

    # Token Density Warning (if any > 1.5)
    max_density = df["Density"].max()
    if max_density > 1.5:
        st.warning(f"⚠️ **High Token Density Detected! (Max: {max_density} tokens/word)**. Your text is 'noisy' (likely code, JSON, or special characters) and will be significantly more expensive to process than standard English.")
    else:
        st.success(f"✅ **Healthy Token Density (Max: {max_density} tokens/word)**. Standard English ratio is ~1.3.")
        

    # Display Metrics in Columns
    st.write("### 2. Context Safety Check")
    cols = st.columns(len(selected_models))
    for idx, row in df.iterrows():
        with cols[idx]:
            st.metric(label=f"{row['Model']} Context", value=f"{row['Tokens']:,} tokens", delta=f"Limit: {row['Context Limit']:,}", delta_color="off")
            fill_pct = row['Context Fill %']
            
            st.progress(fill_pct, text=f"Window Filled: {fill_pct:.1%}")
            
    st.write("---")

    # Layout for charts
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.write("### 3. Tokenizer Efficiency Comparison")
        fig_tokens = px.bar(
            df, x="Model", y="Tokens", color="Tier",
            text="Tokens",
            title="Total Tokens per Model",
            color_discrete_map={"Frontier": "#636EFA", "Budget": "#00CC96", "Legacy": "#EF553B"}
        )
        fig_tokens.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig_tokens.update_layout(yaxis_title="Tokens", uniformtext_minsize=8)
        st.plotly_chart(fig_tokens, use_container_width=True)

    with col_chart2:
        st.write("### 4. Enterprise Scale Cost Projection")
        
        # Melt DataFrame for grouped bar chart
        df_melt = df.melt(id_vars=["Model"], value_vars=["Cost per 1M ($)", "Proj. 10M ($)", "Proj. 100M ($)"], 
                          var_name="Scale", value_name="Cost ($)")
        
        fig_cost = px.bar(
            df_melt, x="Model", y="Cost ($)", color="Scale", barmode="group",
            title="Cost at Different Token Volumes",
            labels={"Cost ($)": "Cost in USD"},
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        # Log scale if max cost is high to make it readable
        if df["Proj. 100M ($)"].max() > 100:
             fig_cost.update_layout(yaxis_type="log")
             fig_cost.update_layout(yaxis_title="Cost in USD (Log Scale)")
        else:
             fig_cost.update_layout(yaxis_title="Cost in USD")
             
        st.plotly_chart(fig_cost, use_container_width=True)

    st.write("### 5. Detailed Cost Strategy Table")
    # Format table beautifully
    display_df = df[["Model", "Tier", "Tokens", "Density", "Cost per 1M ($)", "Actual Payload ($)", "Proj. 10M ($)", "Proj. 100M ($)"]].copy()
    st.dataframe(
        display_df.style.format({
            "Cost per 1M ($)": "${:,.2f}",
            "Actual Payload ($)": "${:,.6f}",
            "Proj. 10M ($)": "${:,.2f}",
            "Proj. 100M ($)": "${:,.2f}",
        }),
        use_container_width=True,
        hide_index=True
    )
elif len(selected_models) == 0:
    st.info("👈 Please select at least one model from the sidebar to continue.")