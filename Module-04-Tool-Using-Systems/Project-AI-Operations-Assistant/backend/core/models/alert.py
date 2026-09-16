from pydantic import BaseModel, Field
from typing import Dict, Optional
from datetime import datetime
from enum import Enum

class Severity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class SystemAlert(BaseModel):
    """
    Represents a production-grade system alert.
    Real-world logs aren't just messages; they are structured data 
    that help engineers (and AI) pinpoint the exact failure point.
    """
    alert_id: str = Field(..., description="Unique identifier for the alert (e.g. UUID)")
    timestamp: datetime = Field(default_factory=datetime.astimezone)
    service_name: str = Field(..., description="The microservice that triggered the alert")
    severity: Severity = Field(..., description="The impact level of the alert")
    error_code: str = Field(..., description="Standardized error code (e.g., SVC_503_TIMEOUT)")
    message: str = Field(..., description="A concise summary of the issue")
    
    # Contextual metadata is crucial in production!
    metadata: Dict[str, str] = Field(
        default_factory=dict, 
        description="Extra context like node_id, region, or request_id"
    )
    
    # Keep the raw data for transparency and debugging
    raw_log: Optional[str] = Field(None, description="The original unstructured log entry")

    class Config:
        json_schema_extra = {
            "example": {
                "alert_id": "ALRT-99210",
                "service_name": "payment-gateway",
                "severity": "CRITICAL",
                "error_code": "PG_504_GATEWAY_TIMEOUT",
                "message": "Upstream bank API timed out after 5000ms",
                "metadata": {
                    "region": "us-east-1",
                    "node_id": "worker-04",
                    "request_id": "req-abcd-1234"
                }
            }
        }
