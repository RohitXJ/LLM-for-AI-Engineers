# Runbook: config-service-info

## Symptom
Service config-service is loading default settings because the user config file was not found.

## Investigation
1. Check if the config file exists in `/etc/config`.
2. Check container logs to verify the warning message.

## Action
This is a standard informational warning. No immediate action is required. Just verify the default settings are acceptable for the current environment using log verification.
