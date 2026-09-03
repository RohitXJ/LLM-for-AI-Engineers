"""
Intent-Based Routing

Uses a lightweight LLM call (or a specialized model) to categorize the
user's intent and return a specific routing decision. This is more flexible
than rule-based routing but introduces a small latency penalty.
"""

from ollama import Client
import json

client = Client(host="http://localhost:11434")
MODEL = "gpt-oss:120b-cloud"

# Define toolsets (like departments)
toolsets = {
    "database": ["query_db", "backup_db"],
    "api": ["get_weather", "send_email"],
    "ops": ["get_server_status", "restart_service"]
}

def route_intent(query: str) -> str:
    """Uses LLM to categorize the intent."""
    prompt = f"""
    Categorize this query into one of these categories: {list(toolsets.keys())}.
    Return ONLY the category name.
    Query: {query}
    """
    response = client.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
    return response.message.content.strip().lower()

user_query = "Please restart the application service."
category = route_intent(user_query)

print(f"User Query: {user_query}")
print(f"Intent Category: {category}")
print(f"Tools to enable: {toolsets.get(category, [])}")
