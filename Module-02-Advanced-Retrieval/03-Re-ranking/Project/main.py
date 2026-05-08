from core import *
CHROMA_LOC = "./cache/chroma_db"
DATA_DIR = "./Data"


def main():
    chroma = Chroma(loc=CHROMA_LOC)
    LLM = Ollama()
    bm25 = KeywordEngine()
    corpus = load_json(DATA_DIR)

    id = []
    for item in corpus:
        id.append(item.get('id'))
        
    fetched_ids = chroma.id_fetch()
    print(fetched_ids)
    new_id = [n_id for n_id in id if n_id not in fetched_ids]
    print(new_id)
    if new_id:
        chroma.data_ingest(new_ids=new_id,corpus=corpus)
        bm25.from_corpus(corpus)
    else:
        print("No New Data Found!")
    #print(corpus)
    #filters = LLM.generate_filter(user_query=corpus[0]['content'],existing_metadata=chroma.get_filter_values())
    #context = chroma.query_context(query_text=corpus[0]['content'], filter_values=filters) #First Retrival MAX 50
    
    

if __name__ == "__main__":
    main()