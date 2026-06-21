from typing import List
from pydantic import BaseModel, Field, ValidationError, field_validator

# 1. Define the target schema for a Research Report
class ResearchReport(BaseModel):
    title: str = Field(description="The title of the research paper")
    summary: str = Field(description="A brief executive summary of the findings")
    key_findings: List[str] = Field(description="At least 3 key findings or takeaways")
    confidence_score: float = Field(description="Confidence score of the analysis between 0.0 and 1.0")

    @field_validator("key_findings")
    @classmethod
    def validate_findings_count(cls, value: List[str]) -> List[str]:
        if len(value) < 3:
            raise ValueError("You must provide at least 3 key findings to ensure depth.")
        return value

    @field_validator("confidence_score")
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        if not (0.0 <= value <= 1.0):
            raise ValueError("Confidence score must be strictly between 0.0 and 1.0")
        return value

# 2. Simulate an LLM that made a mistake on its first attempt
def simulate_imperfect_llm_call() -> dict:
    return {
        "title": "The Impact of Small Language Models in Enterprise",
        "summary": "This report explores how SLMs are replacing larger models for specific tasks.",
        "key_findings": [
            "SLMs reduce inference costs by up to 80%.",
            "Latency is significantly improved."
            # Oops! Only 2 findings instead of 3
        ],
        "confidence_score": 1.2  # Oops! Out of bounds
    }

# 3. Simulate the self-correction prompt generator
def generate_correction_prompt(original_output: dict, errors: List[dict]) -> str:
    error_feedback = ""
    for err in errors:
        field = " -> ".join(str(loc) for loc in err["loc"])
        error_feedback += f"- Field '{field}': {err['msg']}\n"

    prompt = f"""
[SYSTEM NOTICE]
Your previous output failed validation. Please correct the errors and output valid JSON matching the schema.

Original Output:
{original_output}

Validation Errors Found:
{error_feedback}
Please output the corrected JSON object.
"""
    return prompt

def main():
    print("--- Lesson 3.2.4: AI Validation & Self-Correction Loop ---")

    # Step 1: Get the raw output from our simulated LLM
    raw_ai_output = simulate_imperfect_llm_call()
    print("\n[Step 1] Received raw output from LLM...")
    
    # Step 2: Validate the output
    try:
        ResearchReport(**raw_ai_output)
        print("Success on first try!")
    except ValidationError as e:
        print("\n[Step 2] Validation failed! Catching errors...")
        errors = e.errors()
        
        # Step 3: Generate feedback for the LLM
        print("\n[Step 3] Generating self-correction prompt for the LLM:")
        correction_prompt = generate_correction_prompt(raw_ai_output, errors)
        print(correction_prompt)
        
        # Step 4: Simulate the LLM correcting its output based on the feedback
        print("[Step 4] Simulating LLM correction response...")
        corrected_ai_output = {
            "title": "The Impact of Small Language Models in Enterprise",
            "summary": "This report explores how SLMs are replacing larger models for specific tasks.",
            "key_findings": [
                "SLMs reduce inference costs by up to 80%.",
                "Latency is significantly improved.",
                "Fine-tuning SLMs on domain-specific data yields comparable accuracy to frontier models."
            ],
            "confidence_score": 0.95
        }
        
        # Step 5: Re-validate the corrected output
        try:
            validated_report = ResearchReport(**corrected_ai_output)
            print("\n[Success] Corrected output validated perfectly!")
            print(f"Report Title: {validated_report.title}")
            print(f"Key Findings Count: {len(validated_report.key_findings)}")
        except ValidationError as e_sub:
            print("Failed again:", e_sub)

if __name__ == "__main__":
    main()
