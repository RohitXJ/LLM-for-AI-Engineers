"""
Dynamic Tool Dispatcher

Builds a generic tool dispatcher where new tools can be added simply by
registering them. This mirrors how production AI agents scale from a few
tools to dozens without changing the execution logic.
"""

import math
import requests
from datetime import datetime
from ollama import Client

client = Client(host="http://localhost:11434")
MODEL = "gpt-oss:120b-cloud"


class ToolRegistry:
    def __init__(self):
        self.tools = {}

    def register(self, func):
        self.tools[func.__name__] = func
        return func

    def execute(self, name, arguments):
        if name not in self.tools:
            raise ValueError(f"Unknown tool: {name}")

        return self.tools[name](**arguments)

    def all(self):
        return list(self.tools.values())


registry = ToolRegistry()


@registry.register
def get_weather(city: str) -> dict:
    response = requests.get(
        f"https://wttr.in/{city}?format=j1",
        timeout=10,
    )
    response.raise_for_status()

    current = response.json()["current_condition"][0]

    return {
        "city": city,
        "temperature": current["temp_C"],
        "condition": current["weatherDesc"][0]["value"],
    }


@registry.register
def current_time() -> dict:
    now = datetime.now()

    return {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
    }


@registry.register
def calculator(expression: str) -> dict:
    allowed = {
        "__builtins__": {},
        "sqrt": math.sqrt,
        "pow": pow,
        "abs": abs,
        "round": round,
    }

    result = eval(expression, allowed)

    return {
        "expression": expression,
        "result": result,
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
    tools=registry.all(),
)

messages.append(response.message)

while response.message.tool_calls:

    for call in response.message.tool_calls:

        output = registry.execute(
            call.function.name,
            call.function.arguments,
        )

        messages.append(
            {
                "role": "tool",
                "name": call.function.name,
                "content": str(output),
            }
        )

    response = client.chat(
        model=MODEL,
        messages=messages,
    )

    messages.append(response.message)

print("\nAssistant:\n")
print(response.message.content)