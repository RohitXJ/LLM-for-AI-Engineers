"""
Hybrid Retrieval Tool

Demonstrates Hybrid Retrieval by combining keyword search (BM25-like) with
semantic vector search. This mirrors how production RAG systems improve
retrieval quality by leveraging both exact keyword matches and semantic similarity.
"""

from rank_bm25 import BM25Okapi
import chromadb
from ollama import Client

MODEL = "gpt-oss:120b-cloud"

llm = Client(host="http://localhost:11434")


documents = [
    "FastAPI is a modern Python framework for building APIs.",
    "Docker packages applications into lightweight containers.",
    "Ollama enables local execution of large language models.",
    "ChromaDB is a vector database for semantic search.",
    "Hybrid RAG combines keyword retrieval and vector search.",
    "SQLite is a lightweight relational database.",
    "REST APIs allow applications to communicate over HTTP.",
]


tokenized_docs = [doc.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)


chroma = chromadb.Client()
collection = chroma.get_or_create_collection("hybrid_docs")

if collection.count() == 0:
    collection.add(
        ids=[str(i) for i in range(len(documents))],
        documents=documents,
    )


def hybrid_search(query: str, top_k: int = 3) -> dict:
    keyword_results = bm25.get_top_n(
        query.lower().split(),
        documents,
        n=top_k,
    )

    vector_results = collection.query(
        query_texts=[query],
        n_results=top_k,
    )["documents"][0]

    combined = []

    for doc in keyword_results + vector_results:
        if doc not in combined:
            combined.append(doc)

    return {
        "keyword_matches": keyword_results,
        "semantic_matches": vector_results,
        "hybrid_results": combined[:top_k],
    }


messages = [
    {
        "role": "system",
        "content": (
            "Use hybrid_search whenever the user asks questions "
            "about the knowledge base."
        ),
    },
    {
        "role": "user",
        "content": input("Ask: "),
    },
]

response = llm.chat(
    model=MODEL,
    messages=messages,
    tools=[hybrid_search],
)

messages.append(response.message)

while response.message.tool_calls:

    for call in response.message.tool_calls:

        result = hybrid_search(
            **call.function.arguments
        )

        messages.append(
            {
                "role": "tool",
                "name": call.function.name,
                "content": str(result),
            }
        )

    response = llm.chat(
        model=MODEL,
        messages=messages,
    )

    messages.append(response.message)

print("\nAssistant:\n")
print(response.message.content)