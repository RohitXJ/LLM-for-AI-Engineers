"""
Multi API Assistant

Demonstrates an AI assistant that chooses between multiple external REST APIs.
The LLM automatically routes requests to the correct API based on the user's
intent, which is a common architecture in production AI assistants.
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
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def get_random_joke() -> dict:
    try:
        response = requests.get(
            "https://official-joke-api.appspot.com/random_joke",
            timeout=10,
        )

        response.raise_for_status()

        joke = response.json()

        return {
            "setup": joke["setup"],
            "punchline": joke["punchline"],
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def get_exchange_rate(base_currency: str, target_currency: str) -> dict:
    try:
        response = requests.get(
            f"https://open.er-api.com/v6/latest/{base_currency.upper()}",
            timeout=10,
        )

        response.raise_for_status()

        data = response.json()

        if target_currency.upper() not in data["rates"]:
            return {
                "success": False,
                "error": "Unsupported target currency.",
            }

        return {
            "base": base_currency.upper(),
            "target": target_currency.upper(),
            "rate": data["rates"][target_currency.upper()],
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


messages = [
    {
        "role": "system",
        "content": """
You have access to three APIs.

Use get_country_information for:
- countries
- capitals
- population
- geography

Use get_random_joke for:
- jokes
- funny requests

Use get_exchange_rate for:
- currency conversion
- exchange rates
- forex information

Choose the correct tool automatically.
""",
    },
    {
        "role": "user",
        "content": input("Ask: "),
    },
]

response = llm.chat(
    model=MODEL,
    messages=messages,
    tools=[
        get_country_information,
        get_random_joke,
        get_exchange_rate,
    ],
)

messages.append(response.message)

tool_map = {
    "get_country_information": get_country_information,
    "get_random_joke": get_random_joke,
    "get_exchange_rate": get_exchange_rate,
}

while response.message.tool_calls:

    for call in response.message.tool_calls:

        result = tool_map[call.function.name](
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