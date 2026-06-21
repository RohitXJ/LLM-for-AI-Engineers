import json
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Initialize the Ollama LLM
llm = ChatOllama(model="gpt-oss:120b-cloud")

print("--- Demonstrating Error Handling and Validation for Structured Outputs ---")

# Define a simple validation function for a person's data
def validate_person_data(data: dict) -> bool:
    """
    Manually validates if the given dictionary contains required person fields
    and if their types are correct.
    """
    if not isinstance(data, dict):
        print("Validation Error: Input is not a dictionary.")
        return False

    required_keys = {"name", "age", "city"}
    if not required_keys.issubset(data.keys()):
        print(f"Validation Error: Missing required keys. Expected {required_keys}, got {data.keys()}.")
        return False

    if not isinstance(data["name"], str):
        print("Validation Error: 'name' must be a string.")
        return False
    if not isinstance(data["age"], int):
        print("Validation Error: 'age' must be an integer.")
        return False
    if not isinstance(data["city"], str):
        print("Validation Error: 'city' must be a string.")
        return False

    if data["age"] <= 0:
        print("Validation Error: 'age' must be a positive integer.")
        return False

    return True

# Scenario 1: LLM generates (hopefully) valid output
print("\n--- Scenario 1: LLM generates (hopefully) valid output ---")
valid_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert at extracting information and returning it in a structured JSON format. "
        "Your output MUST contain 'name' (string), 'age' (integer), and 'city' (string). "
        "Do not include any other text, only the JSON."
    )),
    ("user", "Extract details for 'Bob, 42 years old, from London'.")
])

chain = valid_prompt | llm
response_valid = chain.invoke({})
print("LLM Raw Output:")
print(response_valid.content)

try:
    parsed_data = json.loads(response_valid.content)
    print("\nParsed JSON:")
    print(json.dumps(parsed_data, indent=2))
    if validate_person_data(parsed_data):
        print("\nSUCCESS: Data parsed and validated successfully!")
    else:
        print("\nFAILURE: Data parsed but failed validation.")
except json.JSONDecodeError as e:
    print(f"\nERROR: Failed to parse JSON from LLM output: {e}")
    print("This indicates the LLM did not provide valid JSON.")
except Exception as e:
    print(f"\nAN UNEXPECTED ERROR OCCURRED: {e}")


# Scenario 2: LLM might generate malformed or invalid output (simulated by a bad prompt)
# NOTE: It can be challenging to reliably make an LLM produce invalid JSON with a single prompt.
# This prompt *attempts* to confuse it, but LLMs are often good at JSON.
print("\n\n--- Scenario 2: LLM might generate malformed/invalid output (simulated) ---")
invalid_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an assistant. Provide details for 'Charlie, age fifty five, from Paris', but deliberately "
        "make the 'age' field a string, not an integer, and output as JSON. "
        "Your output MUST contain 'name' (string), 'age' (string), and 'city' (string). "
        "Do not include any other text, only the JSON."
    )),
    ("user", "Extract details for 'Charlie, age fifty five, from Paris'.")
])

chain = invalid_prompt | llm
response_invalid = chain.invoke({})
print("LLM Raw Output:")
print(response_invalid.content)

try:
    parsed_data = json.loads(response_invalid.content)
    print("\nParsed JSON:")
    print(json.dumps(parsed_data, indent=2))
    if validate_person_data(parsed_data):
        print("\nSUCCESS: Data parsed and validated successfully (against original schema)!")
        print("However, notice the 'age' field's type might be incorrect due to LLM instruction.")
    else:
        print("\nFAILURE: Data parsed but failed validation (as expected for 'age' type).")
except json.JSONDecodeError as e:
    print(f"\nERROR: Failed to parse JSON from LLM output: {e}")
    print("This indicates the LLM did not provide valid JSON, which happens frequently in real-world scenarios.")
except Exception as e:
    print(f"\nAN UNEXPECTED ERROR OCCURRED: {e}")

print("-" * 30)

print("\nExplanation:")
print("03_error_handling_and_validation.py emphasizes the critical need for robust error handling and validation when dealing with LLM outputs.")
print("Even with careful prompting, LLMs can produce malformed JSON or data that doesn't conform to the expected schema (e.g., wrong data types, missing fields).")
print("We use a `try-except json.JSONDecodeError` block to catch cases where the output isn't even valid JSON.")
print("A custom `validate_person_data` function demonstrates how to check for required keys and correct data types, ensuring the extracted data is reliable for downstream applications.")
print("This manual validation is a stepping stone. In Module 3.2, we will introduce Pydantic for more declarative and powerful schema validation.")
