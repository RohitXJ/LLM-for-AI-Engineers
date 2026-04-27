import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

async def dump_to_vector_DB(collection, documents):
    """
    Chunks documents and adds them to the Chroma collection.
    
    Args:
        collection: The ChromaDB collection object (which has an embedding function).
        documents: A list of dicts {"content": str, "metadata": dict}.
    """
    all_ids = []
    all_documents = []
    all_metadatas = []

    # Get the embedding function from the collection
    embedding_fn = collection._embedding_function

    for doc in documents:
        content = doc['content']
        base_metadata = doc['metadata']
        
        # Perform semantic chunking using the embedding function
        chunks = semantic_chunking(embedding_fn, content)
        
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{base_metadata['source']}_p{base_metadata.get('page', 1)}_c{i+1}"
            
            all_ids.append(chunk_id)
            all_documents.append(chunk_text)
            
            # Merge base metadata with chunk-specific metadata
            meta = {**base_metadata, "chunk_index": i + 1}
            all_metadatas.append(meta)

    if all_ids:
        # Chroma will automatically generate embeddings using the collection's embedding_function
        collection.upsert(
            ids=all_ids,
            documents=all_documents,
            metadatas=all_metadatas
        )

def semantic_chunking(embedding_fn, TEXT, percentile_threshold=95):
    """
    Chunks text based on semantic 'jumps' using the provided embedding function.
    """
    # Basic cleanup and split by sentences
    sentences = [s.strip() for s in TEXT.replace('!', '.').replace('?', '.').split('.') if s.strip()]
    
    if len(sentences) < 2:
        return [TEXT]

    # embedding_fn is a Chroma EmbeddingFunction, which takes strings and returns embeddings
    embeddings = embedding_fn(sentences)

    distances = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
        distances.append(1 - sim)
        
    if not distances:
        return [TEXT]
        
    breakpoint_threshold = np.percentile(distances, percentile_threshold)
    
    chunks = []
    current_chunk = [sentences[0]]

    for i, distance in enumerate(distances):
        if distance > breakpoint_threshold:
            chunks.append(". ".join(current_chunk) + ".")
            current_chunk = [sentences[i+1]]
        else:
            current_chunk.append(sentences[i+1])
            
    if current_chunk:
        chunks.append(". ".join(current_chunk) + ".")
        
    return chunks
