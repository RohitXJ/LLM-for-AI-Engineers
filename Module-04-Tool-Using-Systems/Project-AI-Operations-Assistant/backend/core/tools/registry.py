from enum import Enum
import json
import logging
import inspect
from typing import Dict, Any, Callable, List, get_type_hints

class RiskLevel(Enum):
    SAFE = "SAFE"
    DIAGNOSTIC = "DIAGNOSTIC"
    REMEDIATION = "REMEDIATION"

class ToolRegistry:
    """
    A central registry for AI tools. 
    Handles function mapping, HITL security gates, and auto-schema generation.
    """
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}

    def register_tool(self, func: Callable, risk_level: RiskLevel = RiskLevel.SAFE):
        """Adds a function to the registry and auto-generates its schema."""
        name = func.__name__
        self.tools[name] = {
            "func": func,
            "risk_level": risk_level,
            "schema": self._generate_schema(func)
        }
        logging.info(f"Tool '{name}' registered (Risk: {risk_level.value})")

    def _generate_schema(self, func: Callable) -> Dict[str, Any]:
        """Dynamically generates tool schema using introspection."""
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)
        doc = inspect.getdoc(func) or "No description."

        # Map Python types to JSON Schema types
        type_map = {str: "string", int: "integer", float: "number", bool: "boolean"}

        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            # Skip non-LLM arguments
            if param_name in ["collection", "logger", "kwargs"]:
                continue
                
            param_type = type_map.get(type_hints.get(param_name, str), "string")
            properties[param_name] = {"type": param_type}
            
            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        return {
            "name": func.__name__,
            "description": doc.split('\n')[0],
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Returns the auto-generated schemas for all registered tools."""
        return [t["schema"] for t in self.tools.values()]

    def call_tool(self, name: str, args: Dict[str, Any], **kwargs) -> str:
        """
        Intercepts and executes a tool call.
        Enforces HITL only for REMEDIATION tools.
        """
        if name not in self.tools:
            return f"Error: Tool '{name}' not found."

        tool_meta = self.tools[name]
        
        # Security Gate: Only REMEDIATION triggers HITL
        if tool_meta["risk_level"] == RiskLevel.REMEDIATION:
            authorized = self._get_hitl_approval(name, args)
            if not authorized:
                logging.warning(f"HITL: User DENIED execution of {name}")
                return "Error: Human operator denied authorization for this critical action."

        # Execute
        try:
            logging.info(f"Executing tool {name} with args {args}")
            return tool_meta["func"](**args, **kwargs)
        except Exception as e:
            logging.error(f"Tool execution failed ({name}): {e}")
            return f"Error executing tool: {str(e)}"

    def _get_hitl_approval(self, name: str, args: Dict[str, Any]) -> bool:
        """Prompts the console for manual approval."""
        print(f"\n{'='*40}")
        print(f"⚠️  HITL AUTHORIZATION REQUIRED")
        print(f"{'='*40}")
        print(f"Action: {name}")
        print(f"Args:   {json.dumps(args, indent=2)}")
        print(f"{'='*40}")
        
        choice = input("Authorize this action? (y/n): ").strip().lower()
        return choice == 'y'
