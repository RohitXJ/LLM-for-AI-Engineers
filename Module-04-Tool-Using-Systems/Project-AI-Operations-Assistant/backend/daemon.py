from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import time

app = FastAPI()
security = HTTPBearer()

# Shared simulated infrastructure state (Data Plane State)
MOCK_SYSTEM_STATE = {
    "containers": {
        "auth-service": "Exited",
        "database-service": "Exited"
    },
    "ports": {
        5432: "CLOSED",
        8080: "CLOSED"
    }
}

# Simple Auth: In a real system, validate against an environment variable or secret manager
API_TOKEN = "super-secret-daemon-token"

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != API_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")
    return credentials.credentials

@app.get("/status/{service_name}", dependencies=[Depends(verify_token)])
async def get_status(service_name: str):
    status = MOCK_SYSTEM_STATE["containers"].get(service_name, "Exited")
    return {"status": status}

@app.get("/logs/{service_name}", dependencies=[Depends(verify_token)])
async def get_logs(service_name: str, tail: int = 20):
    status = MOCK_SYSTEM_STATE["containers"].get(service_name, "Exited")
    if status == "Up":
        return {"logs": f"Logs for {service_name}: All checks passing."}
    return {"logs": "[2026-09-18 10:00:05] ERROR: Service unreachable (503)\n[2026-09-18 10:01:00] FATAL: Out of memory"}

@app.get("/port/{port}", dependencies=[Depends(verify_token)])
async def check_port(port: int):
    status = MOCK_SYSTEM_STATE["ports"].get(port, "CLOSED")
    return {"port": port, "status": status}

@app.post("/restart/{service_name}", dependencies=[Depends(verify_token)])
async def restart_container(service_name: str):
    # Update mock state
    MOCK_SYSTEM_STATE["containers"][service_name] = "Up"
    if "auth" in service_name:
        MOCK_SYSTEM_STATE["ports"][8080] = "OPEN"
    elif "db" in service_name:
        MOCK_SYSTEM_STATE["ports"][5432] = "OPEN"
        
    return {"message": f"Success: Container '{service_name}' has been restarted and is now 'Up'."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
