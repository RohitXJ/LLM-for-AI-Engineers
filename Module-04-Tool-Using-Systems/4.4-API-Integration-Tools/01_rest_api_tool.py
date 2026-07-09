"""
REST API Tool

Demonstrates how an LLM interacts with a REST API through a dedicated tool.
This forms the foundation for integrating AI systems with external web services.
"""

import requests
from ollama import Client

MODEL = "gpt-oss:120b-cloud"

llm = Client(host="http://localhost:11434")


def get_country_information(country: str) -> dict:
    try:
        response = requests.get(
            f"https://restcountries.com/v3.1/name/{country}",
            timeout=10,
        )

        response.raise_for_status()

        country_data = response.json()[0]

        return {
            "name": country_data["name"]["common"],
            "capital": country_data.get("capital", ["Unknown"])[0],
            "population": country_data["population"],
            "region": country_data["region"],
            "currency": list(
                country_data.get("currencies", {}).keys()
            ),
            "languages": list(
                country_data.get("languages", {}).values()
            ),
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
            "Use get_country_information whenever the user asks "
            "about any country."
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
    tools=[get_country_information],
)

messages.append(response.message)

while response.message.tool_calls:

    for call in response.message.tool_calls:

        result = get_country_information(
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