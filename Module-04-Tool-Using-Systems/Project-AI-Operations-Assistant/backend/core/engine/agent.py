import json
import logging
from typing import List, Dict, Any
from core.models.alert import SystemAlert
from core.models.state import AlertState
from core.models.audit import ActorType, ActionType

class Agent:
    def __init__(self, manager):
        self.manager = manager
        self.logger = manager.logger
        self.client = manager.client
        self.registry = manager.registry
        self.MODEL = manager.MODEL

    def process_alert(self, alert: SystemAlert, trials: int = 10):
        """Orchestrates the reasoning loop to resolve an alert."""
        self.manager.update_state(alert.alert_id, AlertState.INVESTIGATING)
        
        self.logger.info(f"Starting autonomous investigation for alert {alert.alert_id}")
        
        messages = [
            {"role": "system", "content": """You are a senior AI Ops Engineer.
Your goal is to RESOLVE infrastructure alerts using the provided tools.
GUIDELINES:
1. STRICT GROUNDING: Only perform actions described in runbooks or provided as tools. 
2. DIAGNOSE FIRST: Use 'get_container_status', 'get_container_logs', or 'check_service_port' to understand the situation before acting.
3. KNOWLEDGE RETRIEVAL: Use 'search_runbooks' with specific queries (e.g. service name, error code) to find the official resolution steps.
4. REMEDIATION: Only use 'restart_container' if a runbook or diagnostic confirms it's necessary. This is a critical action.
5. NO RAW SHELL: You cannot run arbitrary bash commands. Use only the provided tools.
6. FINAL RESPONSE: Once resolved or failed, provide a structured JSON response:
{"short_summary": "...", "detailed_reasoning": "...", "status": "RESOLVED" or "FAILED"}"""},
            {"role": "user", "content": f"Alert Details:\n{alert.model_dump_json()}"}
        ]
        
        # 2. Main Reasoning Loop
        for attempt in range(trials):
            response = self.client.chat(
                model=self.MODEL, 
                messages=messages, 
                tools=self.registry.get_tool_schemas()
            )
            messages.append(response.message)
            
            # Check for tool calls
            if response.message.tool_calls:
                for tool_call in response.message.tool_calls:
                    name = tool_call.function.name
                    raw_args = tool_call.function.arguments
                    
                    if isinstance(raw_args, str):
                        try:
                            args = json.loads(raw_args)
                        except Exception as e:
                            self.logger.error(f"Failed to parse tool arguments JSON: {e}")
                            args = {}
                    else:
                        args = raw_args or {}
                    
                    self.manager.log_audit(
                        alert_id=alert.alert_id,
                        actor=ActorType.AI,
                        action_type=ActionType.TOOL_CALL,
                        content={"tool": name, "args": args},
                        summary=f"Agent requested tool: {name}"
                    )
                    
                    # Execute tool via Registry (Pass collection for knowledge tool)
                    output = self.registry.call_tool(
                        name, 
                        args, 
                        collection=self.manager.collection
                    )
                    
                    self.manager.log_audit(
                        alert_id=alert.alert_id,
                        actor=ActorType.SYSTEM,
                        action_type=ActionType.TOOL_RESPONSE,
                        content={"tool": name, "response": output},
                        summary=f"Tool {name} result received."
                    )
                    
                    messages.append({
                        "role": "tool",
                        "content": str(output),
                        "name": name
                    })
                # Continue loop to allow agent to process tool output
                continue 
                
            else:
                # Final Resolution - try to parse as JSON
                try:
                    content = response.message.content
                    # Robust JSON extraction
                    if "{" in content and "}" in content:
                        content = content[content.find("{"):content.rfind("}")+1]
                    
                    resolution_data = json.loads(content)
                    short_summary = resolution_data.get("short_summary", "Resolved")
                    status = str(resolution_data.get("status", "RESOLVED")).upper()
                except:
                    # Fallback if not JSON
                    short_summary = "Resolved (Fallback)"
                    status = "RESOLVED"
                    
                self.logger.info(f"Final Resolution ({status}) for {alert.alert_id}: {short_summary}")
                
                self.manager.log_audit(
                    alert_id=alert.alert_id,
                    actor=ActorType.AI,
                    action_type=ActionType.REASONING,
                    content={"resolution": response.message.content},
                    summary=f"Final resolution: {short_summary}"
                )
                
                if status == "RESOLVED":
                    self.manager.update_state(alert.alert_id, AlertState.RESOLVED)
                else:
                    self.manager.update_state(alert.alert_id, AlertState.FAILED)
                break
        else:
            # If loop finished without resolution
            self.manager.update_state(alert.alert_id, AlertState.FAILED)
