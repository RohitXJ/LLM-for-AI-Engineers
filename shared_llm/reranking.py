import os
from typing import List, Dict, Any, Optional
from sentence_transformers import CrossEncoder

class LocalReranker:
    """
    Adapter for running high-precision Cross-Encoder re-ranking locally.
    Uses the BGE-Reranker model by default.
    """
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        """
        Initializes the local re-ranker model.
        
        Args:
            model_name (str): The HuggingFace model path for the Cross-Encoder.
        """
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: List[str], top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Re-scores a list of documents relative to the query.
        
        Args:
            query (str): The search query.
            documents (List[str]): Chunks of text to re-rank.
            top_n (int): Number of top results to return.
            
        Returns:
            List[Dict[str, Any]]: List of dicts with 'text', 'score', and 'original_index'.
        """
        if not documents:
            return []
            
        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs)
        
        results = []
        for idx, score in enumerate(scores):
            results.append({
                "text": documents[idx],
                "score": float(score),
                "index": idx
            })
            
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_n]

class CloudReranker:
    """
    Adapter for Cohere's professional Rerank API.
    Provides industry-leading accuracy via cloud inference.
    """
    def __init__(self, api_key: Optional[str] = None, model: str = "rerank-english-v3.0"):
        """
        Initializes the Cohere client.
        
        Args:
            api_key (Optional[str]): Cohere API Key. Defaults to COHERE_API_KEY env var.
            model (str): The Cohere model to use (e.g., rerank-english-v3.0).
        """
        import cohere
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("CloudReranker requires a COHERE_API_KEY.")
            
        self.client = cohere.ClientV2(api_key=self.api_key)
        self.model = model

    def rerank(self, query: str, documents: List[str], top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Uses Cohere API to re-rank documents.
        
        Args:
            query (str): The search query.
            documents (List[str]): Chunks of text to re-rank.
            top_n (int): Number of results to return.
            
        Returns:
            List[Dict[str, Any]]: Ranked results containing text, score, and index.
        """
        if not documents:
            return []
            
        response = self.client.rerank(
            model=self.model,
            query=query,
            documents=documents,
            top_n=top_n
        )
        
        results = []
        for res in response.results:
            results.append({
                "text": documents[res.index],
                "score": res.relevance_score,
                "index": res.index
            })
            
        return results
