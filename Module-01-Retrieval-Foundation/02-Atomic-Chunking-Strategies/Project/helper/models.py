from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import tiktoken
import warnings
warnings.filterwarnings("ignore")

def recurring_call(chunk_size,chunk_overlap):
    recur_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )
    return recur_splitter

def token_call(chunk_size,chunk_overlap):
    recur_splitter_tiktoken = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4o",
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )
    return recur_splitter_tiktoken

def semantic_call():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

def tokenizer_model_call(model_name):
    return tiktoken.get_encoding(model_name)