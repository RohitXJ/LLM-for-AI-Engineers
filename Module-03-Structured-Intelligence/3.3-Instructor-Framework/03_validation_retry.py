"""
03_validation_retry.py

Topic:
Automatic Validation + Retry

Run this script and observe what happens.
"""

from openai import OpenAI
import instructor
from pydantic import BaseModel, Field

# ---------------- Client ---------------- #

client = instructor.from_openai(
    OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    )
)

MODEL = "gpt-oss:120b-cloud"

# ---------------- Schema ---------------- #

class Student(BaseModel):
    name: str
    age: int = Field(gt=17, lt=30)
    cgpa: float = Field(ge=0, le=10)
    email: str

# ---------------- LLM Call ---------------- #

student = client.chat.completions.create(
    model=MODEL,
    response_model=Student,
    max_retries=3,          # Instructor retries automatically
    messages=[
        {
            "role": "system",
            "content": "Generate realistic student profiles."
        },
        {
            "role": "user",
            "content": (
                "Generate a student who is 12 years old "
                "with a CGPA of 15."
            )
        }
    ]
)

# ---------------- Output ---------------- #

print(student)

print("\nAccess Individual Fields")
print("-" * 30)

print("Name :", student.name)
print("Age  :", student.age)
print("CGPA :", student.cgpa)
print("Email:", student.email)

print("\nJSON")
print(student.model_dump_json(indent=4))