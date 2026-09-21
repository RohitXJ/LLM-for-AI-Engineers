import chromadb
import logging
import json
import os
from pathlib import Path
from typing import Dict, List, Any
from ollama import Client

from core.models.alert import SystemAlert
from core.models.state import AlertState
from core.models.audit import AuditEntry, ActorType, ActionType
from core.tools.registry import ToolRegistry, RiskLevel
from core.tools.knowledge import search_runbooks
from core.tools.system import (
    get_container_status, 
    get_container_logs, 
    check_service_port, 
    restart_container
)

class Manager:
    def __init__(self, logger: logging.Logger, 
                 MODEL: str = "gpt-oss:120b-cloud", 
                 RUNBOOKS: Path = Path("runbooks"),
                 PERSISTENCE_DIR: Path = Path("logs")):
        self.logger = logger
        self.MODEL = MODEL
        self.RUNBOOKS = RUNBOOKS
        self.PERSISTENCE_DIR = PERSISTENCE_DIR
        self.STATE_FILE = self.PERSISTENCE_DIR / "state_tracker.json"
        self.AUDIT_FILE = self.PERSISTENCE_DIR / "audit_log.json"

        # Ensure persistence directory exists
        self.PERSISTENCE_DIR.mkdir(parents=True, exist_ok=True)

        self.logger.info("Initializing Manager and ChromaDB...")
        self.chroma = chromadb.PersistentClient("./database/chroma_data")
        self.collection = self.chroma.get_or_create_collection(name="runBooks")
        self.client = Client(host="http://localhost:11434")

        # Persistence data
        self.states: Dict[str, AlertState] = self._load_states()
        self.audit_log: List[AuditEntry] = self._load_audit()

        # Initialize Tool Registry
        self.registry = ToolRegistry()
        self.registry.register_tool(search_runbooks, risk_level=RiskLevel.DIAGNOSTIC)
        self.registry.register_tool(get_container_status, risk_level=RiskLevel.SAFE)
        self.registry.register_tool(get_container_logs, risk_level=RiskLevel.SAFE)
        self.registry.register_tool(check_service_port, risk_level=RiskLevel.SAFE)
        self.registry.register_tool(restart_container, risk_level=RiskLevel.REMEDIATION)

    def engineStartup(self):
        """Initializes the VDB and ingests new runbooks."""
        from core.database.vector_store import vectorDB
        self.vdb = vectorDB(self.chroma, self.collection, self.RUNBOOKS, self.logger)
        new_files = self.vdb.checkNewFiles()
        
        if new_files:
            self.logger.info(f"Found {len(new_files)} new runbooks. Starting ingestion...")
            self.vdb.ingestNewFiles(new_files)
        else:
            self.logger.info("Knowledge base is up to date.")

    def read_alerts(self, alert_dir: str) -> List[SystemAlert]:
        """Reads, normalizes, and validates alerts from the alerts directory."""
        path = Path(alert_dir)
        validated_alerts = []
        for f in path.glob("*.json"):
            try:
                with open(f, 'r') as file:
                    raw_data = json.load(file)
                    
                    # Normalize: Handle single dict or list of dicts
                    items = raw_data if isinstance(raw_data, list) else [raw_data]
                    
                    for item in items:
                        normalized = self._normalize_alert_data(item)
                        validated_alerts.append(SystemAlert(**normalized))
                        
            except Exception as e:
                self.logger.error(f"Failed to process {f.name}: {e}")
        return validated_alerts

    def _normalize_alert_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Maps messy/varying production keys to our strict SystemAlert schema.
        Example: 'service' -> 'service_name', lowercase severity -> uppercase.
        """
        normalized = data.copy()
        
        # 1. Map Keys
        key_map = {
            "service": "service_name",
            "msg": "message",
            "err_code": "error_code",
            "id": "alert_id"
        }
        for old_key, new_key in key_map.items():
            if old_key in normalized and new_key not in normalized:
                normalized[new_key] = normalized.pop(old_key)

        # 2. Handle Missing Alert ID (Common in raw logs)
        if "alert_id" not in normalized:
            # Generate a deterministic hash-based ID if missing
            import hashlib
            seed = f"{normalized.get('service_name')}{normalized.get('message')}{normalized.get('timestamp')}"
            normalized["alert_id"] = f"AUTO-{hashlib.md5(seed.encode()).hexdigest()[:8]}"

        # 3. Handle Severity Case
        if "severity" in normalized:
            normalized["severity"] = normalized["severity"].upper()

        # 4. Handle Missing Error Code
        if "error_code" not in normalized:
            normalized["error_code"] = "UNKNOWN_ERROR"

        return normalized

    def update_state(self, alert_id: str, new_state: AlertState):
        """Updates the state of an alert and logs the change."""
        old_state = self.states.get(alert_id, "UNKNOWN")
        self.states[alert_id] = new_state
        self.logger.info(f"Alert {alert_id} state changed: {old_state} -> {new_state}")
        
        self.log_audit(
            alert_id=alert_id,
            actor=ActorType.SYSTEM,
            action_type=ActionType.STATE_CHANGE,
            content={"old_state": old_state, "new_state": new_state},
            summary=f"State transitioned to {new_state}"
        )
        self._save_states()

    def log_audit(self, alert_id: str, actor: ActorType, action_type: ActionType, 
                  content: Dict[str, Any], summary: str = None):
        """Adds an entry to the audit log and persists it."""
        entry = AuditEntry(
            alert_id=alert_id,
            actor=actor,
            action_type=action_type,
            content=content,
            summary=summary
        )
        self.audit_log.append(entry)
        self._save_audit()

    # --- Persistence Helpers ---

    def _load_states(self) -> Dict[str, AlertState]:
        if self.STATE_FILE.exists():
            try:
                with open(self.STATE_FILE, "r") as f:
                    data = json.load(f)
                    return {k: AlertState(v) for k, v in data.items()}
            except Exception as e:
                self.logger.error(f"Failed to load states: {e}")
        return {}

    def _save_states(self):
        try:
            with open(self.STATE_FILE, "w") as f:
                json.dump({k: v.value for k, v in self.states.items()}, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save states: {e}")

    def _load_audit(self) -> List[AuditEntry]:
        if self.AUDIT_FILE.exists():
            try:
                with open(self.AUDIT_FILE, "r") as f:
                    data = json.load(f)
                    return [AuditEntry(**entry) for entry in data]
            except Exception as e:
                self.logger.error(f"Failed to load audit log: {e}")
        return []

    def _save_audit(self):
        try:
            with open(self.AUDIT_FILE, "w") as f:
                # Convert Pydantic models to dicts, handling datetimes
                json.dump([json.loads(e.json()) for e in self.audit_log], f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save audit log: {e}")
