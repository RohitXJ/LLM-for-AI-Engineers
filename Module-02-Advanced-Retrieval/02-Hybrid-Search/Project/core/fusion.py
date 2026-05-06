from collections import defaultdict

class HybridRetriever:
    """
    Orchestrates the fusion of keyword and semantic search results using 
    Reciprocal Rank Fusion (RRF). This ensures a balanced ranking that 
    leverages the strengths of both retrieval modes.
    """
    def __init__(self, kw_eng, vec_eng):
        """
        Initializes the HybridRetriever with initialized engines.
        
        Args:
            kw_eng (KeywordEngine): The keyword-based search engine.
            vec_eng (VectorEngine): The vector-based search engine.
        """
        self.kw_eng = kw_eng
        self.vec_eng = vec_eng

    @staticmethod
    def rrf_score(rank, k=60):
        """
        Calculates the Reciprocal Rank Fusion score for a given rank.
        
        Args:
            rank (int): The 1-based rank of the document.
            k (int): A constant to mitigate the impact of low ranks. Default is 60.
            
        Returns:
            float: The RRF score component.
        """
        return 1.0 / (k + rank)

    def search(self, query, top_k=10):
        """
        Executes a hybrid search by performing dual retrieval and fusing the results.
        
        Args:
            query (str): The search query.
            top_k (int): Number of top results to return.
            
        Returns:
            list: A ranked list of tuples (doc_id, fused_score).
        """
        # 1. Execute searches in both engines
        # We fetch more results than top_k to ensure good overlap for fusion
        kw_results = self.kw_eng.search(query, top_k=top_k * 2)
        vec_results = self.vec_eng.search(query, top_k=top_k * 2)

        # 2. Map results to a fused score dictionary
        fused_scores = defaultdict(float)

        # Process Keyword results
        for rank, (doc_id, score) in enumerate(kw_results, start=1):
            fused_scores[doc_id] += self.rrf_score(rank)

        # Process Vector results
        for rank, (doc_id, distance) in enumerate(vec_results, start=1):
            fused_scores[doc_id] += self.rrf_score(rank)

        # 3. Sort by fused score in descending order
        final_results = sorted(
            fused_scores.items(), 
            key=lambda item: item[1], 
            reverse=True
        )

        return final_results[:top_k]
