"""
Module 3.4 - Information Extraction

This example teaches extraction with optional fields and missing information.
The model should extract only what is present and leave unavailable fields
as None instead of hallucinating values.
"""

from typing import Optional

import instructor
from openai import OpenAI
from pydantic import BaseModel, EmailStr

client = instructor.from_openai(
    OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    )
)

MODEL = "gpt-oss:120b-cloud"


class CustomerTicket(BaseModel):
    customer_name: Optional[str] = None
    email: Optional[EmailStr] = None
    product: Optional[str] = None
    issue: str
    priority: Optional[str] = None
    refund_requested: Optional[bool] = None


tickets = [
    """
    Hi, I'm Rohit Gomes.

    My FastAPI course isn't opening after yesterday's update.

    Please help.
    """,

    """
    Hello,

    My name is Alice.

    I purchased the AI Masterclass.

    I was charged twice.

    My email is alice@gmail.com.

    I would like a refund.
    """,

    """
    Urgent!

    Docker Desktop keeps crashing after installation.

    This is affecting our production deployment.
    """,
]

for i, ticket in enumerate(tickets, start=1):

    result = client.chat.completions.create(
        model=MODEL,
        response_model=CustomerTicket,
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract support ticket information. "
                    "Do not invent missing fields. "
                    "Use null when information is unavailable."
                ),
            },
            {
                "role": "user",
                "content": ticket,
            },
        ],
    )

    print(f"\n{'='*20} Ticket {i} {'='*20}")
    print(result)

    print("\nDictionary")
    print(result.model_dump())

    print("\nJSON")
    print(result.model_dump_json(indent=4))