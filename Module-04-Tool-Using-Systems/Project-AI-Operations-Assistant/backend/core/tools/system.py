import httpx
import logging

DAEMON_URL = "http://localhost:8000"
API_TOKEN = "super-secret-daemon-token"

def _call_daemon(method: str, endpoint: str, params: dict = None) -> dict:
    """Helper to perform authenticated requests to the daemon."""
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    url = f"{DAEMON_URL}/{endpoint}"
    try:
        with httpx.Client() as client:
            if method == "GET":
                response = client.get(url, headers=headers, params=params)
            else:
                response = client.post(url, headers=headers)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logging.error(f"Daemon communication error: {e}")
        return {"error": str(e)}

def get_container_status(service_name: str, **kwargs) -> str:
    """Checks if a specific container is running."""
    res = _call_daemon("GET", f"status/{service_name}")
    return f"Status: {res.get('status', 'Unknown')}."

def get_container_logs(service_name: str, tail: int = 20, **kwargs) -> str:
    """Retrieves the last N lines of logs from a container."""
    res = _call_daemon("GET", f"logs/{service_name}", params={"tail": tail})
    return res.get("logs", "Error fetching logs.")

def check_service_port(port: int, **kwargs) -> str:
    """Verifies if a specific network port is listening."""
    res = _call_daemon("GET", f"port/{port}")
    return f"Port {port}: {res.get('status', 'Unknown')}."

def restart_container(service_name: str, **kwargs) -> str:
    """Restarts a stopped or failing container. (REMEDIATION)"""
    res = _call_daemon("POST", f"restart/{service_name}")
    return res.get("message", "Error restarting container.")
