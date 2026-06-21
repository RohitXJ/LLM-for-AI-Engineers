import json
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Initialize the Ollama LLM
llm = ChatOllama(model="gpt-oss:120b-cloud")

print("--- Demonstrating Schema-Driven Generation (via Prompt Engineering) ---")

# Define a desired schema as a Python dictionary structure
# This is our target 'schema' that we want the LLM to follow
person_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "The person's full name"},
        "age": {"type": "integer", "description": "The person's age"},
        "city": {"type": "string", "description": "The city where the person lives"}
    },
    "required": ["name", "age", "city"]
}

# Create a prompt that explicitly tells the LLM to output JSON conforming to the schema
schema_driven_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert at extracting information and returning it in a structured JSON format. "
        "Your output MUST strictly follow this JSON schema:\n"
        f"{json.dumps(person_schema, indent=2)}\n"
        "Do not include any other text, only the JSON."
    )),
    ("user", "Extract the name, age, and city for a person described as 'Alice, 25 years old, from Berlin'.")
], template_format="mustache")

# Invoke the LLM
print("\n--- LLM Output following the schema instructions ---")
chain = schema_driven_prompt | llm
response = chain.invoke({})
print(response.content)

# Attempt to parse the output and perform basic structural validation
try:
    parsed_data = json.loads(response.content)
    print("\nSuccessfully parsed JSON. Now performing basic validation:")

    # Basic manual validation based on expected keys and types
    if all(key in parsed_data for key in ["name", "age", "city"]):
        if isinstance(parsed_data.get("name"), str) and \
           isinstance(parsed_data.get("age"), int) and \
           isinstance(parsed_data.get("city"), str):
            print("Validation successful: Data conforms to the expected structure and types.")
            print(json.dumps(parsed_data, indent=2))
        else:
            print("Validation failed: Data types do not match the schema.")
    else:
        print("Validation failed: Missing required keys in the JSON output.")

except json.JSONDecodeError as e:
    print(f"\nFailed to parse JSON: {e}")
except Exception as e:
    print(f"\nAn unexpected error occurred during validation: {e}")

print("-" * 30)

print("\nExplanation:")
print("02_schema_driven_generation.py demonstrates how to use prompt engineering to guide an LLM to produce JSON output that adheres to a specific schema.")
print("We define the schema structure explicitly in the prompt, instructing the LLM to follow it precisely.")
print("After receiving the LLM's response, we parse it as JSON and perform basic manual checks (keys, types) to validate against our desired structure.")
print("While better than free-form, relying purely on prompt engineering for strict schema adherence can still be brittle. Dedicated tools and libraries are needed for robust validation.")
