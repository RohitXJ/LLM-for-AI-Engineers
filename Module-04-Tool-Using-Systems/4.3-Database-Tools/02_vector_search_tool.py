"""
Vector Search Tool

Demonstrates semantic retrieval using vector embeddings instead of SQL.
The LLM can search documents by meaning, making it suitable for knowledge
bases, documentation, and semantic search applications.
"""

import chromadb
from ollama import Client

MODEL = "gpt-oss:120b-cloud"

llm = Client(host="http://localhost:11434")

db = chromadb.Client()
collection = db.get_or_create_collection("knowledge_base")

documents = [
    {
        "id": "1",
        "text": "FastAPI is a modern Python framework for building high-performance REST APIs.",
    },
    {
        "id": "2",
        "text": "Docker packages applications into portable containers for deployment.",
    },
    {
        "id": "3",
        "text": "Ollama allows large language models to run locally on your own hardware.",
    },
    {
        "id": "4",
        "text": "ChromaDB is a vector database designed for semantic retrieval applications.",
    },
    {
        "id": "5",
        "text": "RAG combines retrieval systems with language models to answer questions using external knowledge.",
    },
]

if collection.count() == 0:
    collection.add(
        ids=[doc["id"] for doc in documents],
        documents=[doc["text"] for doc in documents],
    )


def semantic_search(query: str, top_k: int = 3) -> dict:
    print(f"**Query by AI ** -> {query}")
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
    )

    return {
        "matches": results["documents"][0]
    }

messages = [
    {
        "role": "system",
        "content": (
            "Use semantic_search whenever the user asks questions "
            "about the knowledge base."
            "Do not invent any data or create any new information, answer only from the retrived info."
            "If no suitable info found, respond as no info in the database"
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
    tools=[semantic_search],
)

messages.append(response.message)

while response.message.tool_calls:

    for call in response.message.tool_calls:

        result = semantic_search(
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