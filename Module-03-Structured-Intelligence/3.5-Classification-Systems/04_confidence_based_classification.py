"""
Module 3.5 - Classification Systems

This example teaches confidence-based classification. The LLM predicts
a category along with a confidence score, allowing uncertain predictions
to be flagged for human review.
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


class Sentiment(str, Enum):
    POSITIVE = "Positive"
    NEGATIVE = "Negative"
    NEUTRAL = "Neutral"


class ReviewClassification(BaseModel):
    sentiment: Sentiment
    confidence: float = Field(ge=0, le=1)


reviews = [
    "This laptop is amazing. The battery life is excellent.",
    "The product stopped working after just one day.",
    "The package arrived yesterday. I haven't used it yet.",
    "The camera quality is decent, but the battery could be better."
]


CONFIDENCE_THRESHOLD = 0.75


for i, review in enumerate(reviews, start=1):

    result = client.chat.completions.create(
        model=MODEL,
        response_model=ReviewClassification,
        messages=[
            {
                "role": "system",
                "content": (
                    "Classify the review sentiment as Positive, Negative, or Neutral. "
                    "Return a confidence score between 0 and 1."
                ),
            },
            {
                "role": "user",
                "content": review,
            },
        ],
    )

    print(f"\nReview {i}")
    print("-" * 40)
    print(review)

    print("\nPrediction")
    print(result)

    if result.confidence >= CONFIDENCE_THRESHOLD:
        print("Decision : Auto Accepted")
    else:
        print("Decision : Needs Human Review")