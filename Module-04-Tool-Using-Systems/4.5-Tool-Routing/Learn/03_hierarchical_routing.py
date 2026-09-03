"""
Hierarchical Routing (Departmental)

This is a production-grade pattern. The "Router" delegates to "Specialist" agents.
Each specialist agent has a restricted, highly-focused toolset.
This eliminates confusion and keeps prompt contexts lean.
"""

from ollama import Client

client = Client(host="http://localhost:11434")
MODEL = "gpt-oss:120b-cloud"

# Departments (Specialist Agents)
departments = {
    "finance": {"tools": ["get_exchange_rate", "get_stock_price"], "model": "finance-agent-model"},
    "support": {"tools": ["search_knowledge_base", "create_ticket"], "model": "support-agent-model"}
}

def route_to_department(query: str) -> str:
    """Router agent determines the right department."""
    prompt = f"Which department should handle this query: {query}? Reply with ONLY the department name."
    response = client.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
    return response.message.content.strip().lower()

user_query = "What is the current price of AAPL?"
department = route_to_department(user_query)

# Now, we would call the specific agent for this department
print(f"Router decided: {department}")
print(f"Assigning to Specialist Agent using tools: {departments[department]['tools']}")
