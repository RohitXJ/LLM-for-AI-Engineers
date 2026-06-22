from datetime import date
from typing import List, Optional
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

class WorkExperience(BaseModel):
    company: str = Field(description="Name of the company")
    role: str = Field(description="Job title or role")
    start_date: date = Field(description="Start date of employment (YYYY-MM-DD)")
    end_date: Optional[date] = Field(
        default=None, 
        description="End date of employment (YYYY-MM-DD). Leave empty if currently working here."
    )

    # Model-level validator: validates relationships between multiple fields
    @model_validator(mode="after")
    def validate_dates(self) -> "WorkExperience":
        if self.end_date and self.start_date > self.end_date:
            raise ValueError("start_date cannot be after end_date")
        return self

class Resume(BaseModel):
    candidate_name: str = Field(description="Full name of the candidate")
    contact_email: str = Field(description="Contact email address")
    skills: List[str] = Field(description="List of technical skills")
    experience: List[WorkExperience] = Field(description="Chronological work history")

    # Field-level validator: validates a single field
    @field_validator("contact_email")
    @classmethod
    def validate_email_domain(cls, value: str) -> str:
        if "@" not in value:
            raise ValueError("Invalid email address format")
        return value.lower()

    @field_validator("skills")
    @classmethod
    def ensure_minimum_skills(cls, value: List[str]) -> List[str]:
        if len(value) < 2:
            raise ValueError("A candidate must have at least 2 skills listed")
        return value

def main():
    print("--- Lesson 3.2.3: Custom Validators ---")

    # Valid Resume Data
    valid_resume = {
        "candidate_name": "Aarav Mehta",
        "contact_email": "aarav.mehta@example.com",
        "skills": ["Python", "FastAPI", "LangChain"],
        "experience": [
            {
                "company": "AI Solutions Ltd",
                "role": "Junior AI Engineer",
                "start_date": "2022-06-01",
                "end_date": "2024-01-15"
            }
        ]
    }

    print("\n[1] Parsing valid Resume...")
    try:
        resume = Resume(**valid_resume)
        print(f"Resume for {resume.candidate_name} validated successfully!")
    except ValidationError as e:
        print("Validation failed:", e)

    # Invalid Resume Data (End date before start date)
    invalid_resume = {
        "candidate_name": "Aarav Mehta",
        "contact_email": "aarav.mehta@example.com",
        "skills": ["Python"],  # Only 1 skill (violates minimum skills check)
        "experience": [
            {
                "company": "AI Solutions Ltd",
                "role": "Junior AI Engineer",
                "start_date": "2024-01-15",
                "end_date": "2022-06-01"  # Chronologically impossible!
            }
        ]
    }

    print("\n[2] Parsing invalid Resume...")
    try:
        Resume(**invalid_resume)
    except ValidationError as e:
        print("Validation caught custom business logic errors!")
        for error in e.errors():
            print(f" - Field: {error['loc']} | Error: {error['msg']}")

if __name__ == "__main__":
    main()
