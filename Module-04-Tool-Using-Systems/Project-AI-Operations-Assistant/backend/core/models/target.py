from pydantic import BaseModel, Field
from typing import Optional

class TargetServer(BaseModel):
    """
    Configuration for a single target server infrastructure.
    In Phase 3, this is the connection details for daemon.py.
    """
    id: str = Field(..., description="Unique ID for the server (e.g. node-01)")
    name: str = Field(..., description="Human readable name")
    url: str = Field(..., description="Base URL of the daemon.py (e.g. http://localhost:8000)")
    api_token: str = Field(..., description="Bearer token for daemon auth")
    is_active: bool = True
