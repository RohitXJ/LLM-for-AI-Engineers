from sentence_transformers import SentenceTransformer
import time

models = {
    "Small (384 dims)": "all-MiniLM-L6-v2",
    "Base (768 dims)": "all-mpnet-base-v2"
}

text = "Mastering vector embeddings requires understanding trade-offs between speed and accuracy."

for name, model_name in models.items():
    print(f"\n--- Testing {name} ---")
    model = SentenceTransformer(model_name)
    
    start = time.time()
    embedding = model.encode(text)
    end = time.time()
    
    print(f"Dimensions: {len(embedding)}")
    print(f"Inference Time: {end - start:.4f}s")
    print(f"Memory (Vector Size): {embedding.nbytes} bytes")
