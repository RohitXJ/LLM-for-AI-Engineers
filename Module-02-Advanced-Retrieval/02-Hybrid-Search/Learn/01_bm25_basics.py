import numpy as np
from rank_bm25 import BM25Okapi
from typing import List

# 1. Our small knowledge base
documents = [
    "The iPhone 15 Pro Max has a titanium frame and a high-end camera system.",
    "Samsung Galaxy S24 Ultra features an AI-integrated S-Pen and incredible zoom.",
    "MacBook Pro with M3 Max is the ultimate laptop for creative professionals.",
    "The new Sony headphones have industry-leading noise cancellation technology.",
    "How to fix a 'Connection Timeout' error in your Python database script."
]

# 2. Preprocessing (BM25 expects a list of words/tokens)
tokenized_corpus = [doc.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)

def search_keyword(query: str, top_k: int = 2):
    print(f"\n🔍 Searching for: '{query}'")
    tokenized_query = query.lower().split()
    
    # Get scores for each document
    doc_scores = bm25.get_scores(tokenized_query)
    
    # Get top results
    top_n = bm25.get_top_n(tokenized_query, documents, n=top_k)
    
    # Display scores for transparency
    for i, score in enumerate(doc_scores):
        print(f"Doc {i} Score: {score:.4f} | Content: {documents[i][:50]}...")
    
    return top_n

if __name__ == "__main__":
    # Test 1: Exact Match (High score for 'iPhone')
    search_keyword("iPhone Max")
    
    # Test 2: Rare Keyword (High score for 'Timeout')
    search_keyword("database timeout")
    
    # Test 3: Semantic Concept (BM25 will fail here!)
    # It won't find 'Samsung' or 'iPhone' because it doesn't know what a 'smartphone' is.
    search_keyword("high end smartphone")
