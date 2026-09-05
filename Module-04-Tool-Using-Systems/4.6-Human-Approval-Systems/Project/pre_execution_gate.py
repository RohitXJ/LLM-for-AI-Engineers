from ollama import Client
import json

client = Client(host="http://localhost:11434")
MODEL = "gpt-oss:120b-cloud"

def deploy_service(service_name: str, environment: str) -> dict:
    """Simulate deployment service call."""
    return {"status": "success", "service_name": service_name, "environment": environment}

def verify_tool_access(query: str, args: dict) -> bool:
    if args.get("environment") == "production":
        print("---  VERIFY TOOL ACCESS ---")
        print(f"User Query : {query} \n Args : {args}\n")

        user_input = input("Are you sure to push this deployment to production? (yes/no): ")
        return user_input.strip().lower() == "yes"
    return True

# prompt = "Deploy service 'my-service' to testing environment"
prompt = "Deploy service 'my-service' to production environment"

messages = [{"role": "user", "content": prompt}]

response = client.chat(model=MODEL, messages=messages, tools=[deploy_service])
messages.append(response.message)

if response.message.tool_calls:
    for call in response.message.tool_calls:
        
        args = call.function.arguments
        if isinstance(args, str):
            args = json.loads(args)

        
        if verify_tool_access(query=prompt, args=args):
            tool_out = globals()[call.function.name](**args)
            messages.append({
                "role": "tool", 
                "name": call.function.name, 
                "content": json.dumps(tool_out), 
                "remark": "Authorization Passed"
            })
            print("Committed.")
        else:
            
            messages.append({
                "role": "tool", 
                "name": call.function.name, 
                "content": json.dumps({"status": "failed", "reason": "Halted by security policy"}), 
                "remark": "Authorization Denied by Human"
            })
            print("Rolled back.")

# Finalize
print("\nFinalizing...")
final_response = client.chat(model=MODEL, messages=messages)
messages.append(final_response.message)
print(f"\nAssistant: {final_response.message.content}")
