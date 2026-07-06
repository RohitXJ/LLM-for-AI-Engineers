"""
Input Validation for AI Tools

Shows how production tools validate and sanitize LLM-generated inputs before
execution. This prevents invalid data from reaching external systems and makes
tool execution significantly more reliable.
"""

from typing import Literal
from pydantic import BaseModel, Field, ValidationError
from ollama import Client
import requests

MODEL = "gpt-oss:120b-cloud"
client = Client(host="http://localhost:11434")


class WeatherInput(BaseModel):
    city: str = Field(min_length=2, max_length=50)
    unit: Literal["C", "F"] = "C"


def get_weather(city: str, unit: str = "C") -> dict:
    response = requests.get(
        f"https://wttr.in/{city}?format=j1",
        timeout=10,
    )
    response.raise_for_status()

    current = response.json()["current_condition"][0]

    temp = int(current["temp_C"])

    if unit == "F":
        temp = round((temp * 9 / 5) + 32)

    return {
        "city": city,
        "temperature": temp,
        "unit": unit,
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

while response.message.tool_calls:

    for call in response.message.tool_calls:

        try:
            validated = WeatherInput.model_validate(
                call.function.arguments
            )

            result = get_weather(
                city=validated.city,
                unit=validated.unit,
            )

        except ValidationError as e:
            result = {
                "error": "Input validation failed.",
                "details": e.errors(),
            }

        except Exception as e:
            result = {
                "error": str(e)
            }

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