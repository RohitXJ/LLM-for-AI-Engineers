"""
API Authentication

Demonstrates how AI tools authenticate with APIs using API keys stored in
environment variables. This is the standard approach for securely accessing
most third-party services in production.
"""

import os

import requests
from ollama import Client

MODEL = "gpt-oss:120b-cloud"

llm = Client(host="http://localhost:11434")

API_KEY = os.getenv("OPENWEATHER_API_KEY")


def get_weather(city: str) -> dict:
    if not API_KEY:
        return {
            "success": False,
            "error": "OPENWEATHER_API_KEY environment variable is not set.",
        }

    try:
        response = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={
                "q": city,
                "appid": API_KEY,
                "units": "metric",
            },
            timeout=10,
        )

        response.raise_for_status()

        data = response.json()

        return {
            "success": True,
            "city": data["name"],
            "country": data["sys"]["country"],
            "temperature_c": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "condition": data["weather"][0]["description"],
            "wind_speed": data["wind"]["speed"],
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
            "Use the get_weather tool whenever the user asks "
            "about current weather."
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
    tools=[get_weather],
)

messages.append(response.message)

while response.message.tool_calls:

    for call in response.message.tool_calls:

        result = get_weather(
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