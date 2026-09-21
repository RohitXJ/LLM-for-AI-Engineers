# Runbook: database-service-troubleshooting

## Symptom
Database service is rejecting connections (Connection Refused).

## Investigation
1. Verify if the database container is running.
2. Check if the database service port (5432) is listening.

## Action
If container is down, start the container:
`docker start database-service`
