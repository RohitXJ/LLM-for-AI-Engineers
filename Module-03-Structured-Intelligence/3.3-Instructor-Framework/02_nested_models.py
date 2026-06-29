"""
Module 3.3 - Lesson 2
Topic: Nested Models

Goal:
Learn how Instructor generates objects inside objects.

Employee
 ├── Address
 ├── Company
 └── Skills (List)

Notice that we never parse JSON manually.
"""

from openai import OpenAI
import instructor
from pydantic import BaseModel

# ---------------- Configuration ---------------- #

client = instructor.from_openai(
    OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    )
)

MODEL = "gpt-oss:120b-cloud"

# ---------------- Schemas ---------------- #

class Address(BaseModel):
    city: str
    country: str
    zipcode: int


class Company(BaseModel):
    name: str
    role: str
    experience: int


class Employee(BaseModel):
    name: str
    age: int
    address: Address
    company: Company
    skills: list[str]

# ---------------- LLM Call ---------------- #

employee = client.chat.completions.create(
    model=MODEL,
    response_model=Employee,
    max_retries=5,
    messages=[
        {
            "role": "system",
            "content": "Generate realistic employee profiles."
        },
        {
            "role": "user",
            "content": "Generate an AI Engineer from India."
        }
    ]
)

# ---------------- Results ---------------- #

print("\nEmployee")
print(employee)

print("\nAccess Nested Objects")
print("-" * 30)

print("Name      :", employee.name)
print("City      :", employee.address.city)
print("Country   :", employee.address.country)
print("Company   :", employee.company.name)
print("Role      :", employee.company.role)
print("Experience:", employee.company.experience)

print("\nSkills")
for skill in employee.skills:
    print("•", skill)

print("\nDictionary")
print(employee.model_dump())

print("\nJSON")
print(employee.model_dump_json(indent=4))