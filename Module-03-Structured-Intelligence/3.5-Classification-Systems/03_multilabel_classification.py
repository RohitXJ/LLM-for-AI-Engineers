"""
Module 3.5 - Classification Systems

This example teaches Multi-Label Classification using Instructor + Pydantic.
The LLM can assign multiple labels to a single input.
"""

from enum import Enum

import instructor
from openai import OpenAI
from pydantic import BaseModel


client = instructor.from_openai(
    OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    )
)

MODEL = "gpt-oss:120b-cloud"


class Topic(str, Enum):
    AI = "AI"
    MACHINE_LEARNING = "Machine Learning"
    PYTHON = "Python"
    DOCKER = "Docker"
    FASTAPI = "FastAPI"
    LANGGRAPH = "LangGraph"
    POSTGRESQL = "PostgreSQL"


class ArticleClassification(BaseModel):
    topics: list[Topic]


articles = [
    """
    This tutorial teaches how to build REST APIs using FastAPI
    and PostgreSQL.
    """,

    """
    Learn how to build AI agents using Python, LangGraph
    and LLMs.
    """,

    """
    This guide covers Docker containers for deploying
    Machine Learning applications.
    """
]


for i, article in enumerate(articles, start=1):

    result = client.chat.completions.create(
        model=MODEL,
        response_model=ArticleClassification,
        messages=[
            {
                "role": "system",
                "content": (
                    "Assign all relevant topics from the predefined list. "
                    "Return only the applicable labels."
                ),
            },
            {
                "role": "user",
                "content": article,
            },
        ],
    )

    print(f"\nArticle {i}")
    print("-" * 40)
    print(article.strip())

    print("\nTopics:")
    for topic in result.topics:
        print("-", topic.value)