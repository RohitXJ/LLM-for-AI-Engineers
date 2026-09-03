"""
Static Subset Routing

In a system with many tools, giving the LLM all tools at once causes performance issues.
Static subset routing filters tools based on simple context (e.g., user tags or session state)
to reduce the token load and improve tool selection accuracy.
"""

from ollama import Client

client = Client(host="http://localhost:11434")
MODEL = "gpt-oss:120b-cloud"

# Imagine 20+ tools...
all_tools = ["get_weather", "get_stock", "query_db", "create_issue", "send_email", "...", "get_server_status"]

def get_router_subset(user_context: str):
    """Simple rule-based router."""
    if "data" in user_context or "database" in user_context:
        return ["query_db"]
    elif "infrastructure" in user_context or "server" in user_context:
        return ["get_server_status"]
    else:
        return ["get_weather", "get_stock"] # Default subset

user_query = "Check the database for user logs."
subset = get_router_subset(user_query.lower())

print(f"User Query: {user_query}")
print(f"Selected Tools: {subset}")

# In production, you would only pass `subset` to the client.chat tools parameter.
# response = client.chat(model=MODEL, messages=..., tools=subset)
