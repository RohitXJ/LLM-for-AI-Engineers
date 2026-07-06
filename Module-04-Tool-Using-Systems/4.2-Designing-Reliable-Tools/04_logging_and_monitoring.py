"""
Logging and Monitoring

Demonstrates how AI tools record every execution to a log file. This enables
debugging, auditing, usage analytics, and monitoring in real-world AI systems.
"""

import logging
import time
import requests
from ollama import Client

MODEL = "gpt-oss:120b-cloud"
client = Client(host="http://localhost:11434")

logging.basicConfig(
    filename="tool_logs.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def get_weather(city: str) -> dict:
    start = time.perf_counter()

    logging.info(f"Tool Started | city={city}")

    try:
        response = requests.get(
            f"https://wttr.in/{city}?format=j1",
            timeout=5,
        )
        response.raise_for_status()

        current = response.json()["current_condition"][0]

        duration = round(time.perf_counter() - start, 3)

        logging.info(
            f"Tool Success | city={city} | duration={duration}s"
        )

        return {
            "success": True,
            "city": city,
            "temperature": current["temp_C"],
            "condition": current["weatherDesc"][0]["value"],
            "duration_seconds": duration,
        }

    except Exception as e:
        duration = round(time.perf_counter() - start, 3)

        logging.exception(
            f"Tool Failed | city={city} | duration={duration}s"
        )

        return {
            "success": False,
            "error": str(e),
            "duration_seconds": duration,
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