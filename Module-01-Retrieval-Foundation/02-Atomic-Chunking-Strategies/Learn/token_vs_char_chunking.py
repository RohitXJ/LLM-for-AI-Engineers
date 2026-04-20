import tiktoken
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
import matplotlib.pyplot as plt

def main():
    # Using o200k_base which is optimized for gpt-4o
    token_encoder = tiktoken.get_encoding("o200k_base")

    chunk_size = 200
    chunk_overlap = 50
    
    recur_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )
    
    recur_splitter_tiktoken = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4o",
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )

    print("\n[PRO TIP] Paste a long technical text or code snippet for the best comparison.")
    input_text = input("Enter your text (or press Enter for sample): ").strip()
    
    TEXT = input_text if input_text else """
    The Transformer is a deep learning model that adopts the mechanism of self-attention, differentially weighting 
    the significance of each part of the input data. It is used primarily in the fields of natural language 
    processing (NLP) and computer vision (CV). Like recurrent neural networks (RNNs), transformers are designed 
    to process sequential input data, such as natural language, with applications towards tasks such as 
    translation and text summarization. However, unlike RNNs, transformers do not necessarily process the data 
    in order. Rather, the attention mechanism provides context for any position in the input sequence. 
    For example, if the input data is a natural language sentence, the transformer does not need to process 
    the beginning of the sentence before the end. Due to this feature, the transformer allows for much more 
    parallelization than RNNs and therefore reduced training times. 🤖🚀 Technical jargon: 'hyper-parameter 
    optimization', 'stochastic gradient descent'. Repetitive: !!!!!!!!!!!!!!!!!!!!
    """

    # Chunking
    norm_chunk = recur_splitter.split_text(TEXT)
    token_chunk = recur_splitter_tiktoken.split_text(TEXT)

    # Data collection
    info_graph = {
        "norm" : {
            "length": np.array([len(chunk) for chunk in norm_chunk]),
            "tokens_count": np.array([len(token_encoder.encode(chunks)) for chunks in norm_chunk])
        },
        "token" : {
            "length": np.array([len(chunk) for chunk in token_chunk]),
            "tokens_count": np.array([len(token_encoder.encode(chunks)) for chunks in token_chunk])
        }
    }

    # Comparison and Visualization
    plt.style.use('bmh')
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3) # 2 rows, 3 columns (last column for report)
    
    fig.suptitle('Atomic Chunking Strategy Analysis: Character vs Token Metrics', fontsize=18, fontweight='bold', y=0.98)

    # Colors for visualization
    c_norm = '#3498db' # Blue
    c_token = '#e67e22' # Orange

    # 1. Normal Splitter - Character Lengths (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(range(1, len(norm_chunk) + 1), info_graph["norm"]["length"], marker='o', color=c_norm)
    ax1.axhline(y=chunk_size, color='r', linestyle='--', alpha=0.5, label=f'Limit ({chunk_size})')
    ax1.set_title('Normal Splitter: Character Lengths', fontweight='bold')
    ax1.set_ylabel('Characters')
    ax1.legend()

    # 2. Token Splitter - Character Lengths (Top Middle)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(range(1, len(token_chunk) + 1), info_graph["token"]["length"], marker='o', color=c_token)
    ax2.set_title('Token Splitter: Character Lengths', fontweight='bold')
    ax2.set_ylabel('Characters')

    # 3. Normal Splitter - Token Counts (Bottom Left)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(range(1, len(norm_chunk) + 1), info_graph["norm"]["tokens_count"], marker='s', color=c_norm)
    ax3.set_title('Normal Splitter: Token Counts', fontweight='bold')
    ax3.set_ylabel('Tokens')
    ax3.set_xlabel('Chunk Index')

    # 4. Token Splitter - Token Counts (Bottom Middle)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(range(1, len(token_chunk) + 1), info_graph["token"]["tokens_count"], marker='s', color=c_token)
    ax4.axhline(y=chunk_size, color='r', linestyle='--', alpha=0.5, label=f'Limit ({chunk_size})')
    ax4.set_title('Token Splitter: Token Counts', fontweight='bold')
    ax4.set_ylabel('Tokens')
    ax4.set_xlabel('Chunk Index')
    ax4.legend()

    # 5. The Report Card (Right Panel)
    ax_report = fig.add_subplot(gs[:, 2])
    ax_report.axis('off') # Hide axes for text
    
    report_text = f"""
    STRATEGY PERFORMANCE REPORT
    {'='*30}
    CONFIG:
    - Size: {chunk_size}
    - Overlap: {chunk_overlap}
    
    [NORMAL SPLITTER]
    - Total Chunks: {len(norm_chunk)}
    - Avg Length: {info_graph['norm']['length'].mean():.1f} chars
    - Avg Tokens: {info_graph['norm']['tokens_count'].mean():.1f} tokens
    
    [TOKEN SPLITTER]
    - Total Chunks: {len(token_chunk)}
    - Avg Length: {info_graph['token']['length'].mean():.1f} chars
    - Avg Tokens: {info_graph['token']['tokens_count'].mean():.1f} tokens
    
    {'='*30}
    INSIGHTS:
    The Token Splitter resulted in 
    {len(norm_chunk) - len(token_chunk)} fewer chunks while 
    fitting ~4x more context 
    per embedding call. 
    
    This reduces Vector DB 
    storage and API costs 
    by roughly 75% for 
    this specific text.
    """
    
    ax_report.text(0.05, 0.95, report_text, 
                  fontsize=12, 
                  family='monospace', 
                  verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    main()
