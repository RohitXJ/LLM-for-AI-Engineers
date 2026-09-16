from enum import Enum

class AlertState(str, Enum):
    """
    Formal lifecycle of a production alert.
    This ensures the AI doesn't perform contradictory actions 
    and humans can track progress in real-time.
    """
    NEW = "NEW"                       # Alert just ingested
    INVESTIGATING = "INVESTIGATING"   # AI is currently reasoning/searching runbooks
    AWAITING_APPROVAL = "AWAITING_APPROVAL" # Action proposed, waiting for Human-In-The-Loop
    REMEDIATING = "REMEDIATING"       # Tool is currently executing
    RESOLVED = "RESOLVED"             # Issue fixed and verified
    FAILED = "FAILED"                 # Remediation failed, needs manual intervention
    SNOOZED = "SNOOZED"               # Low priority, ignored for now
