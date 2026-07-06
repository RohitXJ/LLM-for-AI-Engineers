"""
Error Handling for AI Tools

Demonstrates how production tools gracefully handle failures instead of
crashing. The LLM receives structured error responses and can explain
the failure or suggest the next action to the user.
"""

from ollama import Client
import requests

MODEL = "gpt-oss:120b-cloud"
client = Client(host="http://localhost:11434")


def get_weather(city: str) -> dict:
    try:
        response = requests.get(
            f"https://wttr.in/{city}?format=j1",
            timeout=10,
        )

        response.raise_for_status()

        current = response.json()["current_condition"][0]

        return {
            "success": True,
            "city": city,
            "temperature": current["temp_C"],
            "condition": current["weatherDesc"][0]["value"],
        }

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Request timed out.",
            "code": "TIMEOUT",
        }

    except requests.exceptions.HTTPError as e:
        return {
            "success": False,
            "error": str(e),
            "code": "HTTP_ERROR",
        }

    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": str(e),
            "code": "NETWORK_ERROR",
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "code": "UNKNOWN_ERROR",
        }


messages = [
    {
        "role": "system",
        "content": (
            "If a tool returns success=False, explain the error clearly "
            "and suggest what the user should do next."
        ),
    },
    {
        "role": "user",
        "content": input("Ask: "),
    },
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