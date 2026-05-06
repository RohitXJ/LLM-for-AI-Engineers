import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from typing import List

# --- SETUP ---
documents = [
    "The iPhone 15 Pro Max has a titanium frame and a high-end camera system.",
    "Samsung Galaxy S24 Ultra features an AI-integrated S-Pen and incredible zoom.",
    "MacBook Pro with M3 Max is the ultimate laptop for creative professionals.",
    "The new Sony headphones have industry-leading noise cancellation technology.",
    "How to fix a 'Connection Timeout' error in your Python database script."
]

# 1. Setup Keyword Search (BM25)
tokenized_corpus = [doc.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)

# 2. Setup Vector Search (Chroma + SentenceTransformers)
model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.Client()
collection = client.create_collection(name="hybrid_search")
collection.add(
    ids=[f"doc_{i}" for i in range(len(documents))],
    embeddings=model.encode(documents).tolist(),
    documents=documents
)

# --- HYBRID LOGIC ---
def rrf(vector_ranks: List[str], keyword_ranks: List[str], k: int = 60):
    rrf_scores = {}
    for rank, doc in enumerate(vector_ranks):
        rrf_scores[doc] = rrf_scores.get(doc, 0) + 1.0 / (k + rank + 1)
    for rank, doc in enumerate(keyword_ranks):
        rrf_scores[doc] = rrf_scores.get(doc, 0) + 1.0 / (k + rank + 1)
    
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

def hybrid_search(query: str, top_k: int = 2):
    print(f"\n🎯 [HYBRID SEARCH] Query: '{query}'")
    
    # 1. Get Keyword Results
    tokenized_query = query.lower().split()
    keyword_results = bm25.get_top_n(tokenized_query, documents, n=len(documents))
    
    # 2. Get Vector Results
    query_emb = model.encode([query]).tolist()
    vector_raw = collection.query(query_embeddings=query_emb, n_results=len(documents))
    vector_results = vector_raw['documents'][0]
    
    # 3. Fuse them
    fused_results = rrf(vector_results, keyword_results)
    
    # 4. Return top K
    return fused_results[:top_k]

if __name__ == "__main__":
    # Test 1: The "Best of Both Worlds" Test
    # "S24" is a keyword match, "phone" is a semantic match.
    results = hybrid_search("S24 phone", top_k=2)
    for doc, score in results:
        print(f"Final Score: {score:.5f} | {doc}")

    # Test 2: Exact error code
    results = hybrid_search("Connection Timeout", top_k=1)
    print(f"\nTop Result: {results[0][0]}")
