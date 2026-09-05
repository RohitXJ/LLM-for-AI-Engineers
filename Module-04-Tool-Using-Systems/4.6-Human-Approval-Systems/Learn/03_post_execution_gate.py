"""
03_post_execution_gate.py

Verifies the *output* of a tool before the agent finishes.
Crucial for operations tasks where you need to check the result.
"""

from ollama import Client
import json

client = Client(host="http://localhost:11434")
MODEL = "gpt-oss:120b-cloud"

def update_salary(employee_id: int, new_salary: int):
    # Simulate DB change
    return {"status": "success", "new_salary": new_salary}

def verify_result(result: dict) -> bool:
    print(f"\n--- ⚠️  VERIFY RESULT ⚠️ ---")
    print(json.dumps(result, indent=2))
    return input("Approve result? (yes/no): ").lower() == "yes"

messages = [{"role": "user", "content": "Update employee 101 salary to 50000"}]
response = client.chat(model=MODEL, messages=messages, tools=[update_salary])
messages.append(response.message)

if response.message.tool_calls:
    for call in response.message.tool_calls:
        # Execute first
        result = update_salary(**call.function.arguments)
        
        # Verify result
        if verify_result(result):
            messages.append({"role": "tool", "name": call.function.name, "content": str(result)})
            print("Committed.")
        else:
            messages.append({"role": "tool", "name": call.function.name, "content": "Rejected : Didn't pass the verification by human"})
            print("Rolled back.")

# Finalize
final = client.chat(model=MODEL, messages=messages)
print(f"\nAssistant: {final.message.content}")
