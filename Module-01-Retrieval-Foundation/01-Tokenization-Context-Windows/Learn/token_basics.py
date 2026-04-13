import tiktoken

# Load GPT-4 tokenizer
encoding = tiktoken.get_encoding("cl100k_base")
text = "Artificial Intelligence is amazing!"

# Encode & Decode
tokens = encoding.encode(text)
print(f"IDs: {tokens}")

for t_id in tokens:
    print(f"{t_id} -> '{encoding.decode([t_id])}'")
