from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "The cat sits outside",
    "A man is playing guitar",
    "The feline is resting outdoors",
    "I love pepperoni pizza"
]

embeddings = model.encode(sentences)

def compute_score(idx1, idx2):
    # Reshape for sklearn
    vec1 = embeddings[idx1].reshape(1, -1)
    vec2 = embeddings[idx2].reshape(1, -1)
    return cosine_similarity(vec1, vec2)[0][0]

print(f"Similarity (Cat vs Feline): {compute_score(0, 2):.4f}")
print(f"Similarity (Cat vs Guitar): {compute_score(0, 1):.4f}")
print(f"Similarity (Cat vs Pizza): {compute_score(0, 3):.4f}")
