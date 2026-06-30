"""
Module 3.5 - Classification Systems

This example teaches Binary Classification using Instructor + Pydantic.
The LLM must classify the input into one of two predefined categories.
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


class SpamLabel(str, Enum):
    SPAM = "Spam"
    NOT_SPAM = "Not Spam"


class EmailClassification(BaseModel):
    label: SpamLabel
    confidence: float = Field(ge=0, le=1)


emails = [
    "Congratulations! You have won an iPhone. Click here to claim your prize.",
    "Your Amazon order has been shipped and will arrive tomorrow.",
    "Limited-time offer! Earn $5000/day working from home.",
    "Hi Rohit, let's schedule our project meeting for Monday."
]


for i, email in enumerate(emails, start=1):

    result = client.chat.completions.create(
        model=MODEL,
        response_model=EmailClassification,
        messages=[
            {
                "role": "system",
                "content": (
                    "Classify the email as Spam or Not Spam. "
                    "Return a confidence score between 0 and 1."
                ),
            },
            {
                "role": "user",
                "content": email,
            },
        ],
    )

    print(f"\nEmail {i}")
    print("-" * 40)
    print(email)
    print("\nPrediction")
    print(result)