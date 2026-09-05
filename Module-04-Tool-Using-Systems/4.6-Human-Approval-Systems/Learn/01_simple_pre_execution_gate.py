"""
01_simple_pre_execution_gate.py

Manual simulation of a pre-execution gate.
We define a destructive action and wrap it in an approval function.
"""

def destructive_action(data_id: str):
    return f"Data {data_id} deleted."

def approval_gate(action: str, params: dict) -> bool:
    print(f"\n--- ⚠️  ACTION REQUESTED ⚠️ ---")
    print(f"Action: {action}")
    print(f"Params: {params}")
    decision = input("Approve? (yes/no): ").lower()
    return decision == "yes"

# Simulation
if approval_gate("delete_data", {"data_id": "123"}):
    print(destructive_action("123"))
else:
    print("Action blocked.")
