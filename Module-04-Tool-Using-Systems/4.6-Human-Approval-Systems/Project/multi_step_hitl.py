"""Task 2: The "Multi-Step Verification" Chain (multi_step_hitl.py)
  Build an agent that handles a two-stage process: "Transfer Funds" and "Verify Balance."
   1. Tools: transfer_funds(amount: int, account_id: int) and get_balance(account_id: int).
   2. Constraint:
       * Stage 1: The LLM calls transfer_funds. Gate: You must show the planned transfer and get approval.
       * Stage 2: After approval, execute transfer_funds.
       * Stage 3: Immediately call get_balance. Gate: Show the new balance to the human. Ask: "Is the final balance expected?"
   3. Requirements:
       * This requires a loop that manages state between tool calls.
       * You must maintain the conversation history so the LLM understands the result of the balance check in the final confirmation message."""

from ollama import Client
import json

client = Client(host="http://localhost:11434")
MODEL = "gpt-oss:120b-cloud"

def transfer_funds(amount: int, account_id: int):
    """Simulate a transfer of funds to a specified account."""

    print("--- SECURITY CHECK FOR FUND TRANSFER TASKS ---")
    print(f"The agent wants to make a transfer of {amount} to account {account_id}.")
    user_input = input("Do you want to proceed with the transfer? (yes/no) ").lower().strip()
    if user_input == "yes":
        result = "granted"
    else:
        result = "denied"
    return {"status": "success", "authorization": result}

def get_balance(account_id: int):
    """Simulate a balance check for the given account ID."""

    return {"status": "success", "account_id": account_id, "balance": 14530}


messages = [{"role": "user", "content": "Transfer 1000 to account 138457"}]

while True:
    response = client.chat(model=MODEL, messages=messages, tools=[transfer_funds, get_balance])
    messages.append(response.message)
    
    if not response.message.tool_calls:
        print(f"\nAssistant: {response.message.content}")
        break
        
    for call in response.message.tool_calls:
        func_name = call.function.name
        
        # Handle cases where arguments might already be a dict
        args = call.function.arguments
        if isinstance(args, str):
            args = json.loads(args)
        
        print(f"\n--- CALLING TOOL: {func_name} ---")
        
        if func_name == "transfer_funds":
            result = transfer_funds(**args)
            messages.append({
                "role": "tool",
                "name": func_name,
                "content": json.dumps(result)
            })
            #Force balance check to meet the two-stage requirement.
            messages.append({"role": "user", "content": "Now, please call the get_balance tool for account 138457 to verify the new balance."})
            if result.get("authorization") != "granted":
                print("Transfer denied.")
                # We stop the chain if denied
                final_response = client.chat(model=MODEL, messages=messages)
                print(f"\nAssistant: {final_response.message.content}")
                exit()
        
        elif func_name == "get_balance":
            result = get_balance(**args)
            messages.append({
                "role": "tool",
                "name": func_name,
                "content": json.dumps(result)
            })
            print(f"Balance check: {result['balance']}")
            
            # The Gate
            user_verification = input("Is the final balance expected? (yes/no) ").lower().strip()
            messages.append({
                "role": "user",
                "content": f"The human verified the balance. Response: {user_verification}."
            })

