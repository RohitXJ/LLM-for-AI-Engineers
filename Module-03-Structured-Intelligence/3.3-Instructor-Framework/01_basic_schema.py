from openai import OpenAI
import instructor

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

BASE_URL = "http://localhost:11434/v1"
API_KEY = "ollama"

# Change this if needed.
MODEL_NAME = "gpt-oss:120b-cloud"

# ---------------------------------------------------------------------
# Create Instructor Client
# ---------------------------------------------------------------------

client = instructor.from_openai(
    OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
    )
)



class Employee(BaseModel):
    name: str = Field(description="Employee full name")

    age: int = Field(
        description="Employee age in years"
    )

    department: str = Field(
        description="Department where employee works"
    )

    salary: float = Field(
        description="Annual salary in USD"
    )


# ---------------------------------------------------------------------
# Step 2 : Ask the LLM
# ---------------------------------------------------------------------

print("=" * 70)
print("Generating Employee...\n")

employee = client.chat.completions.create(
    model=MODEL_NAME,

    # THIS IS THE MAGIC
    response_model=Employee,

    messages=[
        {
            "role": "system",
            "content": (
                "You generate realistic employee profiles."
            ),
        },
        {
            "role": "user",
            "content": (
                "Generate a fake AI Engineer."
            ),
        },
    ],
)

# ---------------------------------------------------------------------
# Step 3 : Observe the Result
# ---------------------------------------------------------------------

print("Returned Object Type:")
print(type(employee))

print("\nEmployee Object:")
print(employee)

print("\nAccess Like Normal Python Object")
print("--------------------------------")
print("Name       :", employee.name)
print("Age        :", employee.age)
print("Department :", employee.department)
print("Salary     :", employee.salary)

# ---------------------------------------------------------------------
# Pydantic Utilities
# ---------------------------------------------------------------------

print("\nDictionary")
print("----------")
print(employee.model_dump())

print("\nJSON")
print("----")
print(employee.model_dump_json(indent=4))

# ---------------------------------------------------------------------
# Compare with Traditional LLM
# ---------------------------------------------------------------------

print("\n" + "=" * 70)

print(
"""
Without Instructor:

response = client.chat.completions.create(...)

print(response.choices[0].message.content)

↓

{
"name":"John",
"age":"Twenty Five"
}

Now YOU have to:

1. Parse JSON
2. Convert age to integer
3. Handle malformed JSON
4. Handle missing fields
5. Retry if invalid

------------------------------------------------------------

With Instructor:

employee = client.chat.completions.create(
    response_model=Employee,
    ...
)

Done.

Instructor handled everything.
"""
)

print("=" * 70)

