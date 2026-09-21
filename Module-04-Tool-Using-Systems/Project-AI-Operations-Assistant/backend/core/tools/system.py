import logging
import time

# Shared simulated infrastructure state
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

def get_container_status(service_name: str, **kwargs) -> str:
    """
    Checks if a specific container is running.
    """
    status = MOCK_SYSTEM_STATE["containers"].get(service_name, "Exited")
    logging.info(f"Checking status for {service_name}: {status}")
    return f"Status: {status}. Container '{service_name}'."

def get_container_logs(service_name: str, tail: int = 20, **kwargs) -> str:
    """
    Retrieves the last N lines of logs from a container.
    """
    logging.info(f"Fetching logs for: {service_name}")
    status = MOCK_SYSTEM_STATE["containers"].get(service_name, "Exited")
    if status == "Up":
        return f"Logs for {service_name}: All checks passing."
    return "[2026-09-18 10:00:05] ERROR: Service unreachable (503)\n[2026-09-18 10:01:00] FATAL: Out of memory"

def check_service_port(port: int, **kwargs) -> str:
    """
    Verifies if a specific network port is listening.
    """
    status = MOCK_SYSTEM_STATE["ports"].get(port, "CLOSED")
    logging.info(f"Checking port {port}: {status}")
    return f"Port {port}: {status}."

def restart_container(service_name: str, **kwargs) -> str:
    """
    Restarts a stopped or failing container. (REMEDIATION)
    """
    logging.warning(f"REMEDIATION INITIATED: Restarting {service_name}")
    print(f"\n>>> [SIMULATED EXECUTION] docker restart {service_name}")
    
    # Update mock state
    MOCK_SYSTEM_STATE["containers"][service_name] = "Up"
    if "auth" in service_name:
        MOCK_SYSTEM_STATE["ports"][8080] = "OPEN"
    elif "db" in service_name:
        MOCK_SYSTEM_STATE["ports"][5432] = "OPEN"
        
    time.sleep(1)
    return f"Success: Container '{service_name}' has been restarted and is now 'Up'."
