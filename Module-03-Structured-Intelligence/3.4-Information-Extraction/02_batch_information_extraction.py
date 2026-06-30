"""
Module 3.4 - Information Extraction

This example teaches batch information extraction. Multiple unstructured
documents are processed one by one into validated Pydantic objects.
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


class Candidate(BaseModel):
    name: str
    email: Optional[EmailStr] = None
    company: Optional[str] = None
    role: Optional[str] = None
    experience_years: Optional[int] = None
    skills: list[str]


documents = [
    """
    Alice Johnson is a Machine Learning Engineer at Google.
    She has 5 years of experience.
    Skills: Python, TensorFlow, Kubernetes.
    Contact: alice@gmail.com
    """,
    """
    Bob Smith works as a Data Scientist at Microsoft.
    He has two years of experience building NLP systems.
    Skills include Python, PyTorch and Pandas.
    Email: bob.smith@outlook.com
    """,
    """
    Charlie recently joined Amazon as an AI Engineer.
    He has 4 years of industry experience.
    Skills: Docker, FastAPI, LangGraph, PostgreSQL.
    Contact: charlie@amazon.com
    """
]


results = []

for i, document in enumerate(documents, start=1):

    candidate = client.chat.completions.create(
        model=MODEL,
        response_model=Candidate,
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract structured information from the text. "
                    "Do not invent missing information."
                ),
            },
            {
                "role": "user",
                "content": document,
            },
        ],
    )

    results.append(candidate)

    print(f"\nCandidate {i}")
    print(candidate)

print("\nSummary")
print("-" * 40)

for candidate in results:
    print(
        f"{candidate.name:15}"
        f"{candidate.company:12}"
        f"{candidate.experience_years} yrs"
    )