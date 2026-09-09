# Project Roadmap: AI DevOps Reliability Specialist (Production Grade)

## 1. Executive Summary
An autonomous reliability operator designed for secure microservices management. It transforms unstructured observability data into structured, actionable operations, governed by strict HITL security and audit-ready telemetry.

## 2. Infrastructure Architecture
- **Control Plane:** Python-based Agent (Our Project) acting as the brain.
- **Data Plane (Target Servers):** Each target server runs a secure 'Command-Runner Daemon' (a lightweight FastAPI service) that exposes local operational functions as an authenticated REST API.
- **Communication:** Secure, token-authenticated HTTPS requests between Control Plane and Data Plane.

## 3. Detailed Development Phases

### Phase 1: Structured Data & Schema Governance
*Goal: Deterministic ingestion of messy observability data.*
- **Schema Definition:** Implement Pydantic `SystemAlert` with versioning, strictly typed severity levels, and component tagging.
- **Validation Layer:** Build a rigid Parser/Validator service that rejects malformed log entries before they reach the LLM.
- **State Definition:** Define the 'AlertState' (New, Investigating, AwaitingApproval, Remediating, Resolved, Failed).

### Phase 2: RAG Pipeline & Knowledge Management
*Goal: Technical documentation as an operational tool.*
- **Data Prep:** Build a cleaner/ingestor to turn raw markdown Runbooks into structured `TroubleshootingStep` chunks.
- **Vector Indexing:** Set up ChromaDB with persistent storage and metadata-filtering (e.g., filter by service name).
- **Retrieval Evaluation:** Implement an 'Evaluation' step: the agent must retrieve the *correct* runbook for 3 test scenarios before advancing.

### Phase 3: Secure Tooling & Connectivity
*Goal: Safely interacting with the infrastructure.*
- **API Design:** Implement the `Command-Runner Daemon` interface (local REST API).
- **Tool Auth:** Implement token-based authentication for all tool calls (Control Plane -> Daemon).
- **Safety Hardening:** Implement timeouts, request limiters, and atomic operation locks (preventing two actions at once).

### Phase 4: HITL Governance & Audit Trail
*Goal: Regulatory compliance and human oversight.*
- **State Machine:** Implement a formal workflow (using a state machine pattern) to transition between Investigation -> Approval -> Execution -> Verification.
- **Audit Logging:** Every tool request, human approval, and system action must be written to an immutable `audit_log.json` file.
- **Recovery:** Implement state-persistence so the system can recover from a crash at any stage (Investigation, AwaitingApproval, etc.).

### Phase 5: Observability & Console
*Goal: Visibility into the brain.*
- **Telemetry:** Expose the agent’s internal reasoning (Chain-of-Thought) and tool call history to the UI.
- **Feedback Loop:** Build a UI component for manual human intervention/override.
- **Production Readiness:** Add log health-checks to ensure the AI isn't hallucinating infrastructure state.

## 4. Operational Pipeline
1. **Ingestion & Validation:** Log/Alert -> `SystemAlert` Pydantic Model.
2. **Context Enrichment:** `SystemAlert` + Retrieval (Runbooks) -> `DiagnosticReport`.
3. **Reasoning:** `DiagnosticReport` -> LLM Logic -> `ActionPlan`.
4. **Governance:** `ActionPlan` -> HITL Approval Gate -> `Audit Log`.
5. **Execution:** Approved `ActionPlan` -> REST API Tool Call -> Data Plane.
6. **Verification:** Health Check -> Post-Action Report -> `AlertState` Update.

## 5. Industrial Success Metrics
- **Audit Fidelity:** 100% of actions are recorded in the audit log.
- **System Integrity:** 0 unauthorized actions performed.
- **Remediation Speed:** Measurable decrease in MTTD (Mean Time to Diagnose).
- **Failure Handling:** 100% graceful degradation on AI failure (Human handover).
