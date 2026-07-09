"""
External Service Tool

Demonstrates how an AI system integrates with an external SaaS service.
This example creates GitHub issues through the GitHub REST API, a common
automation task used in AI operations and developer assistants.
"""

import os

import requests
from ollama import Client

MODEL = "gpt-oss:120b-cloud"

llm = Client(host="http://localhost:11434")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_OWNER = os.getenv("GITHUB_OWNER")
GITHUB_REPO = os.getenv("GITHUB_REPO")


def create_github_issue(title: str, body: str = "") -> dict:
    if not all([GITHUB_TOKEN, GITHUB_OWNER, GITHUB_REPO]):
        return {
            "success": False,
            "error": (
                "Set GITHUB_TOKEN, GITHUB_OWNER and GITHUB_REPO "
                "environment variables."
            ),
        }

    try:
        response = requests.post(
            f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/issues",
            headers={
                "Authorization": f"Bearer {GITHUB_TOKEN}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            json={
                "title": title,
                "body": body,
            },
            timeout=10,
        )

        response.raise_for_status()

        issue = response.json()

        return {
            "success": True,
            "issue_number": issue["number"],
            "title": issue["title"],
            "url": issue["html_url"],
            "state": issue["state"],
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
            "Use create_github_issue whenever the user asks to "
            "create a GitHub issue."
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
    tools=[create_github_issue],
)

messages.append(response.message)

while response.message.tool_calls:

    for call in response.message.tool_calls:

        result = create_github_issue(
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