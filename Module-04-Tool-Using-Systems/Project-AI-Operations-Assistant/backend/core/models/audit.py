from pydantic import BaseModel, Field
from datetime import datetime
from typing import Any, Dict, Optional
from enum import Enum

class ActorType(str, Enum):
    AI = "AI"
    HUMAN = "HUMAN"
    SYSTEM = "SYSTEM"

class ActionType(str, Enum):
    STATE_CHANGE = "STATE_CHANGE"
    TOOL_CALL = "TOOL_CALL"
    TOOL_RESPONSE = "TOOL_RESPONSE"
    HITL_DECISION = "HITL_DECISION"
    REASONING = "REASONING"

class AuditEntry(BaseModel):
    """
    An immutable record of a single event in the AI's operations lifecycle.
    Essential for regulatory compliance and post-mortem analysis.
    """
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    alert_id: str = Field(..., description="The ID of the alert associated with this entry")
    actor: ActorType = Field(..., description="Who performed the action")
    action_type: ActionType = Field(..., description="What kind of action was performed")
    content: Dict[str, Any] = Field(..., description="Detailed payload of the action")
    summary: Optional[str] = Field(None, description="Human-readable summary of the entry")

    class Config:
        json_schema_extra = {
            "example": {
                "alert_id": "ALRT-123",
                "actor": "AI",
                "action_type": "TOOL_CALL",
                "content": {
                    "tool": "run_command",
                    "args": {"command": "docker restart auth-service"}
                },
                "summary": "Agent requested to restart the auth-service container."
            }
        }
