import pickle
import json
import chromadb
from rank_bm25 import BM25Okapi
from chromadb.utils import embedding_functions

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
        self.doc_ids = [doc['id'] for doc in documents]
        corpus_texts = [f"{doc['title']} {doc['content']}" for doc in documents]
        tokenized_corpus = [self._tokenize(text) for text in corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

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

class VectorEngine:
    """
    Handles semantic search using ChromaDB and dense embeddings.
    This engine captures the 'meaning' of queries beyond simple keyword matching.
    """
    def __init__(self, loc, c_name="Hybrid_Search_SRE"):
        """
        Initializes the ChromaDB client and collection.
        
        Args:
            loc (str): Path to the persistent storage directory.
            c_name (str): Name of the collection.
        """
        self.client = chromadb.PersistentClient(path=loc)
        self.ef = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name=c_name,
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"}
        )
    
    def id_fetch(self):
        """
        Fetches all document IDs currently in the collection.
        
        Returns:
            set: A set of existing document IDs.
        """
        results = self.collection.get(include=[])
        return set(results['ids'])

    def search(self, query, top_k=10):
        """
        Performs a semantic search using cosine similarity.
        
        Args:
            query (str): The search query.
            top_k (int): Number of results to return.
            
        Returns:
            list: A list of tuples (doc_id, distance) ranked by similarity.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        # results['ids'][0] and results['distances'][0] contain the ranked list
        doc_ids = results['ids'][0]
        distances = results['distances'][0]
        
        return list(zip(doc_ids, distances))
