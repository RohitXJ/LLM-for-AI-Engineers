import pickle
import json
from rank_bm25 import BM25Okapi

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

    def from_corpus(self, documents):
        """
        Initializes the BM25 index from a list of documents.
        
        Args:
            documents (list): A list of dictionaries, each containing 'id', 'title', and 'content'.
        """
        self.doc_ids = [doc.get('id') for doc in documents]
        corpus_texts = [f"{doc.get('title', 'N/A')} {doc.get('content', '')}" for doc in documents]
        tokenized_corpus = [self._tokenize(text) for text in corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("BM25 Corpus Built Successfully!")

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
