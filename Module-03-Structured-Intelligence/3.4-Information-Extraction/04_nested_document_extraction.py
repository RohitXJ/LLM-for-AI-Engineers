"""
Module 3.4 - Information Extraction

This example teaches extracting structured data from a complex document with
nested schemas. It demonstrates how real-world documents like resumes or
invoices can be converted into hierarchical Python objects.
"""

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


class Education(BaseModel):
    degree: str
    university: str
    graduation_year: int


class Experience(BaseModel):
    company: str
    role: str
    years: int


class Candidate(BaseModel):
    name: str
    education: Education
    experience: Experience
    skills: list[str]


resume = """
Rahul Sharma completed his Bachelor of Technology from IIT Delhi in 2022.

After graduation he joined NVIDIA as an AI Engineer where he has been
working for 3 years.

His skills include Python, PyTorch, CUDA, Docker, FastAPI and Kubernetes.
"""


candidate = client.chat.completions.create(
    model=MODEL,
    response_model=Candidate,
    messages=[
        {
            "role": "system",
            "content": (
                "Extract structured information from the document. "
                "Do not invent missing information."
            ),
        },
        {
            "role": "user",
            "content": resume,
        },
    ],
)

print(candidate)

print("\nEducation")
print(candidate.education)

print("\nExperience")
print(candidate.experience)

print("\nSkills")
for skill in candidate.skills:
    print("-", skill)

print("\nDictionary")
print(candidate.model_dump())

print("\nJSON")
print(candidate.model_dump_json(indent=4))