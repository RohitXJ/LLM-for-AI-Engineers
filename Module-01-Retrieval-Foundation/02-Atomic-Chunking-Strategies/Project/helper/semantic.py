import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def semantic_chunking(splitter, TEXT, percentile_threshold=95):
    """
    Chunks text based on semantic 'jumps' using adaptive percentile thresholding.
    
    Args:
        splitter: The embedding model.
        TEXT: The raw text content.
        percentile_threshold: The percentile of cosine distances to use as a breakpoint.
                             Higher = fewer, larger chunks. (95th percentile of distance 
                             is the same as the 5th percentile of similarity).
    """
    # 1. Split into sentences (simple newline/period split for demo)
    # Senior Note: In production, use a proper sentence tokenizer like NLTK or Spacy.
    sentences = [s.strip() for s in TEXT.replace('!', '.').replace('?', '.').split('.') if s.strip()]
    
    if len(sentences) < 2:
        return [TEXT]

    # 2. Get embeddings
    embeddings = splitter.encode(sentences)

    # 3. Calculate cosine distances (1 - similarity) between adjacent sentences
    distances = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
        distances.append(1 - sim) # Distance is the opposite of similarity
        
    # 4. Calculate adaptive threshold
    # We look for distances that are in the top X percentile (the biggest "jumps")
    breakpoint_threshold = np.percentile(distances, percentile_threshold)
    
    # 5. Split sentences into chunks
    chunks = []
    current_chunk = [sentences[0]]

    for i, distance in enumerate(distances):
        if distance > breakpoint_threshold:
            # The jump is big enough! Start a new chunk.
            chunks.append(". ".join(current_chunk) + ".")
            current_chunk = [sentences[i+1]]
        else:
            current_chunk.append(sentences[i+1])
            
    # Add the last chunk
    if current_chunk:
        chunks.append(". ".join(current_chunk) + ".")
        
    return chunks
