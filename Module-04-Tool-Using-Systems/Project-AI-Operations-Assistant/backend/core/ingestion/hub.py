import logging
from typing import Dict, Any, List, Optional
from core.ingestion.base import BaseIngestionAdapter
from core.models.alert import SystemAlert

class IngestionHub:
    """
    Central hub for managing alert ingestion plugins.
    It routes incoming raw payloads to the appropriate adapter for normalization.
    """
    def __init__(self):
        self.adapters: Dict[str, BaseIngestionAdapter] = {}
        self.logger = logging.getLogger("IngestionHub")

    def register_adapter(self, adapter: BaseIngestionAdapter):
        """Adds a new adapter to the hub."""
        self.adapters[adapter.source_name] = adapter
        self.logger.info(f"Ingestion adapter registered: {adapter.source_name}")

    def handle_webhook(self, source: str, payload: Any) -> List[SystemAlert]:
        """
        Entry point for webhooks. Routes the payload to the correct adapter.
        Handles both single alerts and lists of alerts (standard in Prometheus).
        """
        if source not in self.adapters:
            self.logger.error(f"No adapter found for source: {source}")
            return []

        adapter = self.adapters[source]
        
        # Prometheus and others often send a list of alerts in one payload
        raw_alerts = []
        if isinstance(payload, list):
            raw_alerts = payload
        elif isinstance(payload, dict) and "alerts" in payload:
            raw_alerts = payload["alerts"]
        else:
            raw_alerts = [payload]

        normalized_alerts = []
        for raw in raw_alerts:
            try:
                alert = adapter.normalize(raw)
                normalized_alerts.append(alert)
            except Exception as e:
                self.logger.error(f"Failed to normalize alert from {source}: {e}")
        
        return normalized_alerts
