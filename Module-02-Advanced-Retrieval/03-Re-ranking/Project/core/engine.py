import pickle
import json
from rank_bm25 import BM25Okapi
from collections import defaultdict

class KeywordEngine:
    """
    Handles keyword-based retrieval using the BM25 algorithm.
    This engine is responsible for tokenizing technical text and calculating relevance scores.
    """
    def __init__(self):
        self.bm25 = None
        self.doc_ids = []

    def _tokenize(self, text):
        """
        Tokenizes input text for technical documentation.
        Converts to lowercase and splits by whitespace.
        
        Args:
            text (str): The text to tokenize.
            
        Returns:
            list: A list of tokens.
        """
        return text.lower().split()

    def from_corpus(self, documents, save_path=None):
        """
        Initializes the BM25 index from a list of documents.
        """
        self.doc_ids = [doc.get('id') for doc in documents]
        corpus_texts = [f"{doc.get('title', 'N/A')} {doc.get('content', '')}" for doc in documents]
        tokenized_corpus = [self._tokenize(text) for text in corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("BM25 Corpus Built Successfully!")
        
        if save_path:
            self.save(save_path)

    def save(self, file_path):
        """
        Saves the BM25 index and doc_ids to a pickle file.
        """
        import os
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump({'bm25': self.bm25, 'doc_ids': self.doc_ids}, f)
        print(f"BM25 Index saved to {file_path}")

    def load(self, file_path):
        """
        Loads the BM25 index and doc_ids from a pickle file.
        """
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                self.bm25 = data['bm25']
                self.doc_ids = data['doc_ids']
            print(f"BM25 Index loaded from {file_path}")
            return True
        except FileNotFoundError:
            print(f"No saved BM25 index found at {file_path}")
            return False
        except Exception as e:
            print(f"Error loading BM25 index: {e}")
            return False

    def search(self, query, top_k=10):
        """
        Performs a keyword search using BM25.
        
        Args:
            query (str): The search query.
            top_k (int): Number of results to return.
            
        Returns:
            list: A list of tuples (doc_id, score) ranked by relevance.
        """
        if not self.bm25:
            return []
        
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Combine IDs and scores, then sort
        results = list(zip(self.doc_ids, scores))
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]


class RFF:
    """
    Orchestrates the fusion of keyword and semantic search results using 
    Reciprocal Rank Fusion (RRF). This ensures a balanced ranking that 
    leverages the strengths of both retrieval modes.
    """
    def __init__(self, kw_eng, vec_eng):
        """
        Initializes the HybridRetriever with initialized engines.
        
        Args:
            kw_eng (KeywordEngine): The keyword-based search engine.(BM25)
            vec_eng (VectorEngine): The vector-based search engine.(Chroma)
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

    def search(self, query,target_ids, top_k=10):
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
        vec_results = self.vec_eng.query_context(query, target_ids=target_ids)

        # 2. Map results to a fused score dictionary
        fused_scores = defaultdict(float)

        # Process Keyword results
        for rank, (doc_id, score) in enumerate(kw_results, start=1):
            fused_scores[doc_id] += self.rrf_score(rank)

        # Process Vector results
        if vec_results and 'ids' in vec_results and vec_results['ids']:
            for rank, doc_id in enumerate(vec_results['ids'][0], start=1):
                fused_scores[doc_id] += self.rrf_score(rank)

        # 3. Sort by fused score in descending order
        final_results = sorted(
            fused_scores.items(), 
            key=lambda item: item[1], 
            reverse=True
        )

        return final_results[:top_k]
