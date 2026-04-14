import tiktoken
from transformers import AutoTokenizer

def models_call(model_name:str)->object:
    """
    Returns the encoder/tokenizer object for the given code_name.
    - Uses tiktoken for OpenAI's base encodings.
    - Uses AutoTokenizer for Hugging Face models (Llama, DeepSeek).
    """
    # Define which strings belong to the tiktoken library
    TIKTOKEN_ENCODINGS = ["o200k_base", "cl100k_base", "p50k_base", "r50k_base"]

    try:
        if model_name in TIKTOKEN_ENCODINGS:
            # Returns a tiktoken.Encoding object
            return tiktoken.get_encoding(model_name)
        else:
            # Returns a Hugging Face FastTokenizer object
            # Note: trust_remote_code=True is required for models like DeepSeek
            return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
    except Exception as e:
        print(f"Failed to load encoder for {model_name}: {e}")
        return None