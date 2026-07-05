"""
Manual Tool Calling

Demonstrates the core idea behind function calling by allowing an LLM to decide
when to invoke a weather tool and then generating a final answer using the tool output.
"""

import json
import requests
from ollama import Client

client = Client(host="http://localhost:11434")
MODEL = "ornith_1:9B-q4"


def get_weather(city: str):
    response = requests.get(
        f"https://wttr.in/{city}?format=j1",
        timeout=10,
    )
    response.raise_for_status()

    data = response.json()
    current = data["current_condition"][0]

    return {
        "city": city,
        "temperature_c": current["temp_C"],
        "feels_like_c": current["FeelsLikeC"],
        "humidity": current["humidity"],
        "weather": current["weatherDesc"][0]["value"],
    }


tools = {
    "get_weather": get_weather,
}

tool_schema = {
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string"
            }
        },
        "required": ["city"]
    }
}

user_query = input("Ask: ")

system_prompt = f"""
You are a tool-using AI.

Available Tool:
{json.dumps(tool_schema, indent=2)}

If the tool is needed, respond ONLY in JSON:

{{
    "tool": "get_weather",
    "arguments": {{
        "city": "<city>"
    }}
}}

If no tool is required:

{{
    "tool": null,
    "response": "<answer>"
}}
"""

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_query},
]

response = client.chat(
    model=MODEL,
    messages=messages,
)

decision = json.loads(response.message.content)

if decision["tool"] is None:
    print("\nAssistant:")
    print(decision["response"])
else:
    result = tools[decision["tool"]](**decision["arguments"])

    final_messages = [
        {
            "role": "system",
            "content": "Answer the user using the provided tool result.",
        },
        {
            "role": "user",
            "content": user_query,
        },
        {
            "role": "tool",
            "content": json.dumps(result),
        },
    ]

    final = client.chat(
        model=MODEL,
        messages=final_messages,
    )

    print("\nTool Output:")
    print(json.dumps(result, indent=2))

    print("\nAssistant:")
    print(final.message.content)