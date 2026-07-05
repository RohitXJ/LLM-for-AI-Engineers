"""
Multiple Tool Selection

Demonstrates how an LLM chooses between multiple available tools based on
the user's request. This is the foundation of agentic systems that can
interact with different software capabilities.
"""

import requests, json
from datetime import datetime
from ollama import Client

client = Client(host="http://localhost:11434")
MODEL = "gpt-oss:120b-cloud"


def get_weather(city: str) -> dict:
    response = requests.get(
        f"https://wttr.in/{city}?format=j1",
        timeout=10,
    )
    response.raise_for_status()

    current = response.json()["current_condition"][0]

    return {
        "city": city,
        "temperature_c": current["temp_C"],
        "condition": current["weatherDesc"][0]["value"],
    }


def get_current_time() -> dict:
    now = datetime.now()

    return {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "weekday": now.strftime("%A"),
    }


def add_numbers(a: float, b: float) -> dict:
    return {
        "result": a + b
    }


available_tools = {
    "get_weather": get_weather,
    "get_current_time": get_current_time,
    "add_numbers": add_numbers,
}

messages = [
    {
        "role": "user",
        "content": input("Ask: "),
    }
]

response = client.chat(
    model=MODEL,
    messages=messages,
    tools=[
        get_weather,
        get_current_time,
        add_numbers,
    ],
)
old_response = response
messages.append(response.message)

while response.message.tool_calls:

    for tool in response.message.tool_calls:

        function = available_tools[tool.function.name]

        result = function(**tool.function.arguments)

        messages.append(
            {
                "role": "tool",
                "name": tool.function.name,
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
print("\n")
print(json.dumps(dict(old_response), indent=4, default=str))