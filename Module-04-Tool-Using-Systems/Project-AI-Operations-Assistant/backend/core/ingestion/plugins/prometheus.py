from typing import Dict, Any
from core.ingestion.base import BaseIngestionAdapter
from core.models.alert import SystemAlert, Severity

class PrometheusAdapter(BaseIngestionAdapter):
    """
    Adapter for Prometheus Alertmanager webhooks.
    Alertmanager sends a list of alerts. This adapter handles a single alert object
    from the 'alerts' list in the Prometheus payload.
    """
    @property
    def source_name(self) -> str:
        return "prometheus"

    def normalize(self, payload: Dict[str, Any]) -> SystemAlert:
        # Map Prometheus labels to our schema
        labels = payload.get("labels", {})
        annotations = payload.get("annotations", {})
        
        import hashlib
        seed = f"{labels.get('service', 'unknown')}{labels.get('alertname', 'alert')}"
        alert_id = payload.get("fingerprint", f"PROM-{hashlib.md5(seed.encode()).hexdigest()[:8]}")

        return SystemAlert(
            alert_id=alert_id,
            service_name=labels.get("service", labels.get("job", "unknown-prometheus-job")),
            severity=self._map_severity(labels.get("severity", "warning")),
            error_code=labels.get("alertname", "PROMETHEUS_ALERT"),
            message=annotations.get("description", annotations.get("summary", "No description provided.")),
            metadata={**labels, **annotations}
        )

    def _map_severity(self, prom_severity: str) -> Severity:
        mapping = {
            "critical": Severity.CRITICAL,
            "error": Severity.ERROR,
            "warning": Severity.WARNING,
            "info": Severity.INFO
        }
        return mapping.get(prom_severity.lower(), Severity.ERROR)
