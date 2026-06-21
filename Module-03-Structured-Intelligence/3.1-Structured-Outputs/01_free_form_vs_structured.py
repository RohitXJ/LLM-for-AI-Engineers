import json
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Initialize the Ollama LLM with the specified model
llm = ChatOllama(model="gpt-oss:120b-cloud")

print("--- Demonstrating Free-Form Generation ---")

# Prompt for free-form output
free_form_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "Tell me about John Doe, who is 30 years old and lives in New York.")
])

# Invoke the LLM
print("\n--- Free-Form Output ---")
chain = free_form_prompt | llm
response = chain.invoke({})
print(response.content)
print("-" * 30)


print("\n--- Demonstrating JSON Mode (via Prompt Engineering) ---")

# Prompt for structured output using JSON mode
json_mode_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Your output MUST be in JSON format."),
    ("user", "Provide details for a person named John Doe, who is 30 years old and lives in New York. Include 'name', 'age', and 'city' fields.")
])

# Invoke the LLM
print("\n--- JSON Mode Output (attempted) ---")
chain = json_mode_prompt | llm
response_json = chain.invoke({})
print(response_json.content)

# Attempt to parse the JSON output
try:
    parsed_json = json.loads(response_json.content)
    print("\nSuccessfully parsed JSON:")
    print(json.dumps(parsed_json, indent=2))
except json.JSONDecodeError as e:
    print(f"\nFailed to parse JSON: {e}")
    print("This highlights the challenge of relying solely on prompt engineering for strict JSON adherence.")

print("-" * 30)

print("\nExplanation:")
print("01_free_form_vs_structured.py demonstrates the difference between asking an LLM for free-form text versus attempting to guide it towards JSON output using prompt engineering.")
print("Free-form output is conversational and flexible but inconsistent for programmatic use.")
print("Prompting for JSON helps, but the LLM might still deviate, making strict parsing difficult without explicit validation or a dedicated structured output API.")
