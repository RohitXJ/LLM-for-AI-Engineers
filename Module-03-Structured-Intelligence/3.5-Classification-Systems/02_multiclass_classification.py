"""
Module 3.5 - Classification Systems

This example teaches Multi-Class Classification using Instructor + Pydantic.
The LLM must choose exactly one category from several predefined classes.
"""

from enum import Enum

import instructor
from openai import OpenAI
from pydantic import BaseModel, Field


client = instructor.from_openai(
    OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    )
)

MODEL = "gpt-oss:120b-cloud"


class TicketCategory(str, Enum):
    BILLING = "Billing"
    TECHNICAL = "Technical"
    SALES = "Sales"
    ACCOUNT = "Account"


class TicketClassification(BaseModel):
    category: TicketCategory
    confidence: float = Field(ge=0, le=1)


tickets = [
    "I was charged twice for my subscription.",
    "The application crashes whenever I click the login button.",
    "Can someone give me pricing details for the enterprise plan?",
    "I forgot my password and cannot access my account."
]


for i, ticket in enumerate(tickets, start=1):

    result = client.chat.completions.create(
        model=MODEL,
        response_model=TicketClassification,
        messages=[
            {
                "role": "system",
                "content": (
                    "Classify the support ticket into exactly one of these "
                    "categories: Billing, Technical, Sales, Account. "
                    "Return a confidence score between 0 and 1."
                ),
            },
            {
                "role": "user",
                "content": ticket,
            },
        ],
    )

    print(f"\nTicket {i}")
    print("-" * 40)
    print(ticket)
    print("\nPrediction")
    print(result)