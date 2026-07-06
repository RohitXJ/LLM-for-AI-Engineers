"""
Timeouts and Automatic Retries

Demonstrates how production AI tools recover from temporary network failures
using request timeouts, exponential backoff, and retry logic before reporting
an error to the LLM.
"""

import time
import requests
from ollama import Client

MODEL = "gpt-oss:120b-cloud"
client = Client(host="http://localhost:11434")


def get_weather(city: str) -> dict:
    retries = 3
    delay = 1

    for attempt in range(1, retries + 1):

        try:
            response = requests.get(
                f"https://wttr.in/{city}?format=j1",
                timeout=3,
            )

            response.raise_for_status()

            current = response.json()["current_condition"][0]

            return {
                "success": True,
                "city": city,
                "temperature": current["temp_C"],
                "condition": current["weatherDesc"][0]["value"],
                "attempt": attempt,
            }

        except (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
        ):

            if attempt == retries:
                return {
                    "success": False,
                    "error": "Maximum retry attempts reached.",
                    "attempts": retries,
                }

            time.sleep(delay)
            delay *= 2

        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": str(e),
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

while response.message.tool_calls:

    for call in response.message.tool_calls:

        result = get_weather(**call.function.arguments)

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