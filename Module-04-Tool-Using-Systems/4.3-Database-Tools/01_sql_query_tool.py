"""
SQL Database Query Tool

Demonstrates how an LLM can safely query a SQL database using a dedicated tool.
This is the foundation for AI assistants that retrieve structured business data
instead of relying solely on model knowledge.
"""

import sqlite3, json
from ollama import Client

MODEL = "gpt-oss:120b-cloud"
client = Client(host="http://localhost:11434")


def initialize_database():
    conn = sqlite3.connect("employees.db")
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
        "INSERT INTO employees(name, department, salary) VALUES (?, ?, ?)",
        [
            ("Alice", "Engineering", 85000),
            ("Bob", "Marketing", 65000),
            ("Charlie", "HR", 55000),
            ("David", "Engineering", 91000),
            ("Eva", "Finance", 72000),
        ],
    )

    conn.commit()
    conn.close()


initialize_database()


def query_employee_database(sql_query: str) -> dict:
    sql = sql_query.strip()

    if not sql.lower().startswith("select"):
        return {
            "success": False,
            "error": "Only SELECT queries are allowed.",
        }

    try:
        conn = sqlite3.connect("employees.db")
        conn.row_factory = sqlite3.Row

        cursor = conn.cursor()
        cursor.execute(sql)

        rows = [dict(row) for row in cursor.fetchall()]

        conn.close()

        return {
            "success": True,
            "rows": rows,
            "count": len(rows),
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


messages = [
    {
        "role": "system",
        "content": (
            "Use the SQL tool whenever the user asks about employees. "
            "Generate only SELECT statements."
            "If empty result or count = 0 is returned from tool, it means the query has 0 result"
        ),
    },
    {
        "role": "user",
        "content": input("Ask: "),
    },
]

response = client.chat(
    model=MODEL,
    messages=messages,
    tools=[query_employee_database],
)

messages.append(response.message)

while response.message.tool_calls:

    for call in response.message.tool_calls:

        result = query_employee_database(
            **call.function.arguments
        )

        messages.append(
            {
                "role": "tool",
                "name": call.function.name,
                "content": str(result),
            }
        )

    response = client.chat(
        model=MODEL,
        messages=messages,
    )

    messages.append(response.message)

print("\nAssistant:\n")
print(response.message.content)

print("\n--- Complete Conversation History ---")
print(json.dumps(messages, indent=4, default=str))
print("\n--- Complete LLM Conversation History ---")
print(json.dumps(dict(response), indent=4, default=str))
