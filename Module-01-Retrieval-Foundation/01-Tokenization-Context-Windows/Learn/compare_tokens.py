import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")

def analyze(text, label):
    tokens = encoding.encode(text)
    print(f"\n[{label}]")
    print(f"Text: {text}")
    print(f"Tokens: {len(tokens)}")
    print(f"IDs: {tokens}")

# Case 1: Clean vs Noisy
analyze("Python is a programming language.", "Clean")
analyze("Pyth0n i$ a pr0gramm1ng languag3.", "Noisy")

# Case 2: Standard vs Fragmented
analyze("Hello World", "Simple")
analyze("H_e_l_l_o _W_o_r_l_d", "Fragmented")
