from fastapi import FastAPI, HTTPException, Security, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging
from pathlib import Path

from core.manager import Manager
from core.engine.agent import Agent

# Setup logging
logging.basicConfig(
    filename='logs/ai.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
api_logger = logging.getLogger("ControlPlaneAPI")

app = FastAPI(title="AI Ops Control Plane API", version="2.0")
security = HTTPBearer()

# Control Plane Secret Token (In production, use ENV vars)
CONTROL_PLANE_TOKEN = "super-secret-control-token"

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != CONTROL_PLANE_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid Control Plane token")
    return credentials.credentials

# Initialize Manager & Agent on startup
manager = Manager(api_logger)
manager.engineStartup()
agent = Agent(manager)

@app.post("/api/v1/webhook/{source}", dependencies=[Security(verify_token)])
async def receive_alert_webhook(source: str, payload: dict, background_tasks: BackgroundTasks):
    """
    Universal Webhook Endpoint.
    Routes incoming payloads to the appropriate Ingestion Plugin (Prometheus, Daemon, Generic),
    normalizes them into SystemAlerts, and queues them for the Agent.
    """
    api_logger.info(f"Received webhook from source: {source}")
    
    # Pass through Ingestion Hub
    alerts = manager.hub.handle_webhook(source, payload)
    
    if not alerts:
        raise HTTPException(status_code=400, detail=f"Failed to normalize alert from source '{source}'")

    processed_ids = []
    for alert in alerts:
        processed_ids.append(alert.alert_id)
        # Run agent in background task to avoid blocking the webhook response
        background_tasks.add_task(agent.process_alert, alert)

    return {
        "status": "queued",
        "source": source,
        "processed_alerts": processed_ids,
        "message": f"Successfully ingested {len(alerts)} alert(s). Investigation queued."
    }

@app.get("/api/v1/states", dependencies=[Security(verify_token)])
async def get_system_states():
    """Returns the current state of all tracked alerts."""
    return {"states": manager.states}

@app.get("/api/v1/audit", dependencies=[Security(verify_token)])
async def get_audit_logs(limit: int = 50):
    """Returns the immutable audit trail."""
    return {"audit_log": [json.loads(e.json()) for e in manager.audit_log[-limit:]]}

if __name__ == "__main__":
    import uvicorn
    import json
    uvicorn.run(app, host="0.0.0.0", port=8001)
