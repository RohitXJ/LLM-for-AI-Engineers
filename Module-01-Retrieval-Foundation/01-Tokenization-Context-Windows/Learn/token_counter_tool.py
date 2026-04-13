import tiktoken

# 1. Load the tokenizer used by GPT-4 (cl100k_base)
encoding = tiktoken.get_encoding("cl100k_base")
text = input("Enter the text you want the tokens to be counted off\n : ")
length = len(text.split())

# 2. Text -> Token IDs (Encoding)
tokens = encoding.encode(text)
count = len(tokens)

# 3. Feedback
print(f"Total Words: {length}")
print(f"Total Tokens: {count}")
print(f"Total Cost: ${(count/40)*0.28:.2f} ($0.28 per 40 tokens)")
if count > 200:
    print("Text exceeding the model's context window size!!")
