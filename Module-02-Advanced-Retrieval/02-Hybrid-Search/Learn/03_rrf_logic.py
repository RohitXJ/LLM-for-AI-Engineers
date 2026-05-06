from typing import List, Dict

def reciprocal_rank_fusion(vector_results: List[str], keyword_results: List[str], k: int = 60):
    """
    Combines two ranked lists using Reciprocal Rank Fusion.
    
    Args:
        vector_results: List of document strings (ranked by vector search)
        keyword_results: List of document strings (ranked by BM25)
        k: Smoothing constant (default 60)
    """
    rrf_scores = {}

    # Process Vector Results
    for rank, doc in enumerate(vector_results):
        if doc not in rrf_scores:
            rrf_scores[doc] = 0
        # RRF formula: 1 / (k + rank)
        # Note: rank starts at 0, so we use rank + 1
        rrf_scores[doc] += 1.0 / (k + (rank + 1))

    # Process Keyword Results
    for rank, doc in enumerate(keyword_results):
        if doc not in rrf_scores:
            rrf_scores[doc] = 0
        rrf_scores[doc] += 1.0 / (k + (rank + 1))

    # Sort documents by their RRF scores in descending order
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_docs

if __name__ == "__main__":
    # EXAMPLE SCENARIO:
    # Query: "Python error timeout"
    
    # Vector search found these (good semantic match, but maybe generic)
    v_results = ["Doc A: Python script error", "Doc B: Connection issues", "Doc C: General coding"]
    
    # Keyword search found these (exact match for 'timeout')
    k_results = ["Doc D: Timeout error guide", "Doc A: Python script error", "Doc E: Latency fix"]

    print("🚀 Running Reciprocal Rank Fusion...")
    final_results = reciprocal_rank_fusion(v_results, k_results)

    for doc, score in final_results:
        print(f"Score: {score:.5f} | {doc}")

    # EXPLANATION:
    # Doc A will likely be #1 because it appeared in both lists.
    # Doc D might be #2 because it was #1 in keywords.
