import httpx
import json

URL = "http://localhost:8001/api/v1/webhook/prometheus"
TOKEN = "super-secret-control-token"

payload = {
    "labels": {
        "service": "auth-service",
        "severity": "critical",
        "alertname": "SVC_503_TIMEOUT"
    },
    "annotations": {
        "summary": "Service auth-service is unreachable (503 Service Unavailable)"
    }
}

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

print(f"Sending test webhook to {URL}...")
response = httpx.post(URL, json=payload, headers=headers)
print(f"Status Code: {response.status_code}")
print(f"Response: {response.json()}")
