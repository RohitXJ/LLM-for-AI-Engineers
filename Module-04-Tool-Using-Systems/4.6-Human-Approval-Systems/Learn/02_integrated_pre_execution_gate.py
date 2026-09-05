"""
02_integrated_pre_execution_gate.py

Integrates the approval gate with Ollama.
The agent decides to call a tool, we intercept the call before execution.
"""

from ollama import Client

client = Client(host="http://localhost:11434")
MODEL = "gpt-oss:120b-cloud"

def delete_user(user_id: int):
    return f"User {user_id} deleted."

def approval_gate(tool_name: str, args: dict) -> bool:
    print(f"\n--- ⚠️  LLM WANTS TO CALL: {tool_name} ⚠️ ---")
    print(f"Args: {args}")
    return input("Approve? (yes/no): ").lower() == "yes"

messages = [{"role": "user", "content": "Delete user 55"}]

response = client.chat(model=MODEL, messages=messages, tools=[delete_user])
messages.append(response.message)

if response.message.tool_calls:
    for call in response.message.tool_calls:
        if approval_gate(call.function.name, call.function.arguments):
            # Execute
            result = globals()[call.function.name](**call.function.arguments)
            print(f"Executed: {result}")
        else:
            print("Blocked.")
