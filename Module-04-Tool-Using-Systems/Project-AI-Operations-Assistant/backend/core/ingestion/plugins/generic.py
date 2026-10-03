from typing import Dict, Any
from core.ingestion.base import BaseIngestionAdapter
from core.models.alert import SystemAlert, Severity

class GenericAdapter(BaseIngestionAdapter):
    """
    A catch-all adapter for simple JSON webhooks.
    """
    @property
    def source_name(self) -> str:
        return "generic"

    def normalize(self, payload: Dict[str, Any]) -> SystemAlert:
        # Use existing manager normalization logic via a placeholder
        # In a real system, this would be highly flexible
        return SystemAlert(
            service_name=payload.get("service", payload.get("service_name", "unknown")),
            severity=Severity(payload.get("severity", "ERROR").upper()),
            error_code=payload.get("error_code", "GENERIC_WEBHOOK"),
            message=payload.get("message", "No message provided."),
            metadata=payload.get("metadata", {})
        )
