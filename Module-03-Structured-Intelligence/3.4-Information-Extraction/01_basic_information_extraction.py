"""
Module 3.4 - Information Extraction

This example teaches how to convert unstructured text into a structured
Python object using Instructor + Pydantic. It demonstrates the basic
information extraction pipeline used in production AI systems.
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
    email: EmailStr
    phone: Optional[str] = None
    location: str
    company: str
    role: str
    experience_years: int
    skills: list[str]


document = """
Rohit Gomes is currently working as an AI Engineer at Hexa Global.
He lives in Kolkata, West Bengal.

He has 3 years of experience building AI systems.

His core skills include Python, FastAPI, Docker,
PyTorch, PostgreSQL, LangGraph and Ollama.

You can contact him at
rohit.gomes@gmail.com
or +91 9876543210.
"""


candidate = client.chat.completions.create(
    model=MODEL,
    response_model=Candidate,
    messages=[
        {
            "role": "system",
            "content": (
                "Extract structured information from the given text. "
                "Do not invent any information."
            ),
        },
        {
            "role": "user",
            "content": document,
        },
    ],
)

print(candidate)

print("\nIndividual Fields")
print("-" * 30)
print("Name       :", candidate.name)
print("Email      :", candidate.email)
print("Phone      :", candidate.phone)
print("Location   :", candidate.location)
print("Company    :", candidate.company)
print("Role       :", candidate.role)
print("Experience :", candidate.experience_years)
print("Skills     :", candidate.skills)

print("\nAs Dictionary")
print(candidate.model_dump())

print("\nAs JSON")
print(candidate.model_dump_json(indent=4))