"""Task 1: The Multi-Agent Router (Based on 03_hierarchical_routing.py)
  Create a file router_assessment.py.
   1. Define three toolsets: WeatherTools, CalculatorTools, KnowledgeBaseTools.
   2. Implement a router function that accepts a user query and returns which toolset should be active.
   3. Write a simulation loop:
       - Input: A user query.
       - Output: The Router's decision, and the list of tools assigned to that query."""

from ollama import Client

client = Client(host="http://localhost:11434")
MODEL = "gpt-oss:120b-cloud"

tools = {
    "weather_tools" : {
        "tools" : ["get_weather", "get_temperature"],
        "model" : "Weather-Agent"
        },
    "calculator_tools" : {
        "tools" : ["add", "sub"],
        "model" : "Math-Agent"
        },
    "knowledge_baseTools" : {
        "tools" : ["fetch_data", "ingest_data"],
        "model" : "KB-Agent"
        },
}

def toolset_router(query:str,)->dict:
    """Router function to accept user query and return the needed tools for the query"""
    prompt = f"Based on this user query : {query} What toolset should be used?Available toolsets category: {tools.keys()}Reply with only the toolset category name"
    response = client.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
        )
    return response.message.content.strip().lower()

toolset_out = toolset_router("What is the weather in New York?")
print(f"\nToolset chosen by router {tools[toolset_out]}")