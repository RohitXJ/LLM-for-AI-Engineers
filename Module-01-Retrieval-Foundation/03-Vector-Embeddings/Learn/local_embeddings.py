from sentence_transformers import SentenceTransformer
import numpy as np

# Use a lightweight, popular local model
model = SentenceTransformer('all-MiniLM-L6-v2')

text = "Vector embeddings are the backbone of RAG."
embedding = model.encode(text)

print(f"Text: {text}")
print(f"Vector Type: {type(embedding)}")
print(f"Vector Shape: {embedding.shape}")
print(f"First 5 dimensions: {embedding[:5]}")
