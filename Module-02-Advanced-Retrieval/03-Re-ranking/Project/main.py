import time
from core import *

CHROMA_LOC = "./cache/chroma_db"
DATA_DIR = "./Data"
BM25_CACHE = "./cache/bm25/bm25.pkl"

def main():
    chroma = Chroma(loc=CHROMA_LOC)
    LLM = Ollama()
    bm25 = KeywordEngine()
    corpus = load_json(DATA_DIR)

    ids = [item.get('id') for item in corpus]
    fetched_ids = chroma.id_fetch()
    new_ids = [n_id for n_id in ids if n_id not in fetched_ids]
    
    if new_ids:
        print(f"[*] Found {len(new_ids)} new documents. Synchronizing database...")
        chroma.data_ingest(new_ids=new_ids, corpus=corpus)
        bm25.from_corpus(corpus, save_path=BM25_CACHE)
    else:
        if not bm25.load(BM25_CACHE):
            bm25.from_corpus(corpus, save_path=BM25_CACHE)
    
    rff = RFF(kw_eng=bm25, vec_eng=chroma)
    reranker = CrossEnc()
    
    print("\n" + "="*50)
    print("      ADVANCED POLICY SEARCH AGENT (RRF + BGE)")
    print("="*50)
    print("Type 'exit' or 'quit' to end the session.")
    
    while True:
        query = input("\n[User]: ").strip()
        if not query: continue
        if query.lower() in ['bye', 'exit', 'quit']:
            print("\n[System]: Session ended. Goodbye!")
            break
        
        # --- PHASE 1: Initial Retrieval (Metadata + Vector) ---
        start_t1 = time.perf_counter()
        filters = LLM.generate_filter(user_query=query, existing_metadata=chroma.get_filter_values())
        context_payload = chroma.query_context(query_text=query, filter_values=filters) 
        target_ids = get_ids(context=context_payload)
        end_t1 = time.perf_counter()
        
        if not target_ids:
            print("[System]: No relevant documents found in the initial search.")
            continue

        # --- PHASE 2: Hybrid RRF Search ---
        start_t2 = time.perf_counter()
        result_ids = rff.search(query=query, target_ids=target_ids, top_k=10)
        candidates = filter_context_by_search(context=context_payload, search_results=result_ids)
        end_t2 = time.perf_counter()
        
        # --- PHASE 3: Cross-Encoder Re-ranking ---
        start_t3 = time.perf_counter()
        top_docs = reranker.rank(query=query, candidates=candidates, top_k=2)
        end_t3 = time.perf_counter()
        
        # --- Final Generation ---
        final_context = "\n\n".join(top_docs)
        LLM.answer_question(context=final_context, query=query)

        # --- Performance Metrics ---
        print("-" * 30)
        print(f"⏱️  Performance Metrics:")
        print(f"   1. Metadata + Vector Search: {(end_t1 - start_t1):.4f}s")
        print(f"   2. Hybrid RRF Search:        {(end_t2 - start_t2):.4f}s")
        print(f"   3. Cross-Encoder Re-rank:   {(end_t3 - start_t3):.4f}s")
        print("-" * 30)

    


    

    
    

if __name__ == "__main__":
    main()