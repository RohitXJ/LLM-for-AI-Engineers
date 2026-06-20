import pickle
import os
from typing import List, Tuple, Dict, Optional, Any
from rank_bm25 import BM25Okapi
from collections import defaultdict

class KeywordEngine:
    """
    Handles keyword-based retrieval using the BM25 algorithm.
    Essential for matching specific terminology, IDs, and error codes.
    """
    def __init__(self):
        """Initializes an empty BM25 engine."""
        self.bm25: Optional[BM25Okapi] = None
        self.doc_ids: List[str] = []

    def _tokenize(self, text: str) -> List[str]:
        """Simple technical tokenizer: lowercase and whitespace split."""
        return text.lower().split()

    def build_index(self, documents: List[Dict[str, Any]], save_path: Optional[str] = None) -> None:
        """
        Creates a BM25 index from a list of document dictionaries.
        
        Args:
            documents (List[Dict[str, Any]]): List of dicts containing 'content' and 'metadata'.
            save_path (Optional[str]): If provided, serializes the index to this path.
        """
        self.doc_ids = []
        corpus_texts = []
        
        for i, doc in enumerate(documents):
            # Extract ID from metadata or fallback to index
            metadata = doc.get('metadata', {})
            doc_id = metadata.get('id') or str(i)
            self.doc_ids.append(doc_id)
            
            # Extract searchable text (Title/Source + Content)
            title = metadata.get('title') or metadata.get('source') or ""
            content = doc.get('content', '')
            corpus_texts.append(f"{title} {content}")
            
        tokenized_corpus = [self._tokenize(text) for text in corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        if save_path:
            self.save(save_path)

    def save(self, file_path: str) -> None:
        """Serializes the engine state to a pickle file."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump({'bm25': self.bm25, 'doc_ids': self.doc_ids}, f)

    def load(self, file_path: str) -> bool:
        """Loads a serialized index from a pickle file."""
        if not os.path.exists(file_path):
            return False
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                self.bm25 = data['bm25']
                self.doc_ids = data['doc_ids']
            return True
        except Exception as e:
            print(f"Error loading BM25 index: {e}")
            return False

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Performs keyword search and returns ranked IDs and scores.
        
        Args:
            query (str): The search query.
            top_k (int): Number of results to return.
            
        Returns:
            List[Tuple[str, float]]: List of (doc_id, score) sorted by relevance.
        """
        if not self.bm25:
            return []
        
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        results = list(zip(self.doc_ids, scores))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

class HybridFusion:
    """
    Implements Reciprocal Rank Fusion (RRF) to merge rankings from multiple search engines.
    """
    @staticmethod
    def rrf_score(rank: int, k: int = 60) -> float:
        """Calculates the RRF score component for a specific rank."""
        return 1.0 / (k + rank)

    @classmethod
    def fuse(cls, keyword_results: List[Tuple[str, Any]], vector_results: List[str], top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Merges keyword and vector rankings using Reciprocal Rank Fusion.
        
        Args:
            keyword_results (List[Tuple[str, Any]]): Ranked (ID, score) from BM25.
            vector_results (List[str]): Ordered IDs from Vector Search.
            top_k (int): Number of fused results to return.
            
        Returns:
            List[Tuple[str, float]]: Fused ranked list of (doc_id, fused_score).
        """
        fused_scores = defaultdict(float)

        # Process Keyword ranks
        for rank, (doc_id, _) in enumerate(keyword_results, start=1):
            fused_scores[doc_id] += cls.rrf_score(rank)

        # Process Vector ranks
        for rank, doc_id in enumerate(vector_results, start=1):
            fused_scores[doc_id] += cls.rrf_score(rank)

        # Sort and return
        final_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return final_results[:top_k]
