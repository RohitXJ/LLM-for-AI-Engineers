"""
04_information_extraction.py

Topic:
Information Extraction using Instructor + Pydantic

Convert unstructured text into structured data.
"""

from openai import OpenAI
import instructor
from pydantic import BaseModel, EmailStr
from typing import Optional

# ---------------- Client ---------------- #

client = instructor.from_openai(
    OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    )
)

MODEL = "gpt-oss:120b-cloud"

# ---------------- Schema ---------------- #

class Candidate(BaseModel):
    name: str
    email: EmailStr
    phone: Optional[str]
    location: str
    current_company: str
    experience_years: int
    skills: list[str]

# ---------------- Unstructured Text ---------------- #

resume = """
Rohit Gomes is an AI Engineer currently working at OpenAI India.
He lives in Kolkata, West Bengal.

He has around 3 years of experience building AI systems.

His primary skills include Python, FastAPI, Docker,
PyTorch, LangGraph, Ollama and PostgreSQL.

You can contact him at
rohit.gomes@gmail.com
or +91 9876543210.
"""

# ---------------- Extraction ---------------- #

candidate = client.chat.completions.create(
    model=MODEL,
    response_model=Candidate,
    messages=[
        {
            "role": "system",
            "content": (
                "Extract structured information from the given text. "
                "Only fill fields that can be inferred."
            )
        },
        {
            "role": "user",
            "content": resume
        }
    ]
)

# ---------------- Output ---------------- #

print(candidate)

print("\nFields")
print("-" * 30)

print("Name      :", candidate.name)
print("Email     :", candidate.email)
print("Phone     :", candidate.phone)
print("Location  :", candidate.location)
print("Company   :", candidate.current_company)
print("Experience:", candidate.experience_years)

print("\nSkills")
for skill in candidate.skills:
    print("-", skill)

print("\nDictionary")
print(candidate.model_dump())

print("\nJSON")
print(candidate.model_dump_json(indent=4))