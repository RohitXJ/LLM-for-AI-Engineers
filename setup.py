from setuptools import setup, find_packages

setup(
    name="shared_llm",
    version="0.1.1",
    packages=find_packages(),
    install_requires=[
        "chromadb",
        "langchain-ollama",
        "langchain-core",
        "pydantic",
        "rank_bm25",
        "sentence-transformers",
        "cohere",
        "python-dotenv",
        "langchain-text-splitters"
    ],
    description="A shared library for RAG components in the LLM-for-AI-Engineers roadmap.",
    author="Gemini CLI",
)
