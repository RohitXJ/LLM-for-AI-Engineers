from core import *
CHROMA_LOC = "./cache/chroma_db"
DATA_DIR = "./Data"


def main():
    chroma = Chroma(loc=CHROMA_LOC)
    LLM = Ollama()
    corpus = load_json(DATA_DIR)
    id = []
    metadata = []
    doc = []
    for item in corpus:
        doc.append(f"TITLE: {item.get('title', 'N/A')} CONTENT: {item.get('content', '')}")
        metadata.append({ 
            "department": item.get('department'), 
            "category": item.get('category'), 
            "year": item.get('year') 
        })
        id.append(item.get('id'))
    
    chroma.data_ingest(
        ids=id,
        documents=doc,
        metadatas=metadata
    )
    #print(corpus)
    filters = LLM.generate_filter(user_query=corpus[0]['content'],existing_metadata=chroma.get_filter_values())
    
    context = chroma.query_context(query_text=corpus[0]['content'], filter_values=filters) #First Retrival MAX 50
    
    

if __name__ == "__main__":
    main()