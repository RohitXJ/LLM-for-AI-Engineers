# Runbook: auth-service-restart

## Symptom
Service auth-service is reporting 503 errors.

## Investigation
1. Check container logs.
2. Check resource utilization (CPU/Memory).

## Action
If memory is saturated, restart the service:
`docker restart auth-service`
