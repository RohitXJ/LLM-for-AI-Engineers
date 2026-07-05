"""
Native Function Calling

Demonstrates modern tool calling using Ollama's native tools interface.
The LLM decides whether to call a function, the application executes it,
and the result is returned back to the model for a final response.
"""

import requests
from ollama import Client

client = Client(host="http://localhost:11434")
MODEL = "ornith_1:9B-q4"


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
        "feels_like_c": current["FeelsLikeC"],
        "humidity": current["humidity"],
        "condition": current["weatherDesc"][0]["value"],
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
    tools=[get_weather],
)

messages.append(response.message)

if response.message.tool_calls:
    for tool in response.message.tool_calls:
        result = globals()[tool.function.name](**tool.function.arguments)

        messages.append(
            {
                "role": "tool",
                "name": tool.function.name,
                "content": str(result),
            }
        )

    final = client.chat(
        model=MODEL,
        messages=messages,
    )

    print("\nAssistant:\n")
    print(final.message.content)

else:
    print("\nAssistant:\n")
    print(response.message.content)