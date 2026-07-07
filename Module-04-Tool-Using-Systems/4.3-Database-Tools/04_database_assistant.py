"""
Database Assistant

Demonstrates an AI assistant that dynamically selects between a SQL database
and a vector database. This mirrors how production AI systems route user
queries to the most appropriate retrieval backend.
"""

import sqlite3

import chromadb
from ollama import Client

MODEL = "gpt-oss:120b-cloud"

llm = Client(host="http://localhost:11434")


def initialize_database():
    conn = sqlite3.connect("company.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS employees(
            id INTEGER PRIMARY KEY,
            name TEXT,
            department TEXT,
            salary INTEGER
        )
    """)

    cursor.execute("DELETE FROM employees")

    cursor.executemany(
        """
        INSERT INTO employees(name, department, salary)
        VALUES (?, ?, ?)
        """,
        [
            ("Alice", "Engineering", 85000),
            ("Bob", "Marketing", 65000),
            ("Charlie", "HR", 56000),
            ("David", "Engineering", 91000),
            ("Eva", "Finance", 72000),
        ],
    )

    conn.commit()
    conn.close()


initialize_database()


documents = [
    "FastAPI is a Python framework for building APIs.",
    "Docker packages applications into portable containers.",
    "ChromaDB stores vector embeddings for semantic retrieval.",
    "RAG combines retrieval systems with language models.",
    "SQLite is a lightweight SQL database.",
    "Ollama allows LLMs to run locally.",
]

chroma = chromadb.Client()
collection = chroma.get_or_create_collection("knowledge_base")

if collection.count() == 0:
    collection.add(
        ids=[str(i) for i in range(len(documents))],
        documents=documents,
    )


def sql_search(sql_query: str) -> dict:
    if not sql_query.strip().lower().startswith("select"):
        return {
            "success": False,
            "error": "Only SELECT statements are allowed.",
        }

    try:
        conn = sqlite3.connect("company.db")
        conn.row_factory = sqlite3.Row

        cursor = conn.cursor()
        cursor.execute(sql_query)

        rows = [dict(row) for row in cursor.fetchall()]

        conn.close()

        return {
            "success": True,
            "rows": rows,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def vector_search(query: str, top_k: int = 3) -> dict:
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
    )

    return {
        "matches": results["documents"][0],
    }


messages = [
    {
        "role": "system",
        "content": """
You have two database tools.

Use sql_search for:
- employee data
- salaries
- departments
- IDs
- structured records

Use vector_search for:
- AI concepts
- documentation
- software knowledge
- semantic search

Always choose the correct tool.
""",
    },
    {
        "role": "user",
        "content": input("Ask: "),
    },
]

response = llm.chat(
    model=MODEL,
    messages=messages,
    tools=[
        sql_search,
        vector_search,
    ],
)

messages.append(response.message)

tool_map = {
    "sql_search": sql_search,
    "vector_search": vector_search,
}

while response.message.tool_calls:

    for call in response.message.tool_calls:

        result = tool_map[call.function.name](
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