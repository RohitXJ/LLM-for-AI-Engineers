from pydantic import BaseModel, Field

class ReasoningResponse(BaseModel):
    """
    A structured final response from the AI Agent.
    This enables clean UI rendering while keeping logs searchable.
    """
    short_summary: str = Field(..., description="A one-sentence summary of the action taken.")
    detailed_reasoning: str = Field(..., description="Full technical explanation, log analysis, and outcome.")
    status: str = Field(..., description="The final outcome (e.g., 'RESOLVED', 'FAILED', 'NEEDS_HUMAN')")
