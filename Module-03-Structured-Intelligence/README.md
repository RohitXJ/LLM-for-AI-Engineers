# Module 03: Structured Intelligence

Welcome to **Module 03: Structured Intelligence**. This module focuses on the transition from unstructured data to structured, type-safe information using Pydantic and Large Language Models (LLMs).

---

## 📖 Overview
In modern AI engineering, extracting reliable data from unstructured sources (like resumes, emails, or reports) is critical. This module provides the fundamentals of using Pydantic for data validation and demonstrates how to build a production-grade engine to enforce these structures on LLM outputs.

## 📂 Project Structure

### 1. Pydantic Fundamentals
Located in `3.2-Pydantic-Fundamentals/`, these files cover the core concepts of data validation:
*   **`03_custom_validators.py`**: Demonstrates the use of `@model_validator` to enforce complex business logic (e.g., ensuring `start_date` precedes `end_date` in employment records).
*   **`04_ai_validation_simulation.py`**: Shows how to handle "imperfect" LLM output, illustrating the necessity of strict schemas to catch out-of-bounds values or missing data fields.

### 2. Project: AI Document Intelligence Engine
Located in `Project-AI-Document-Intelligence-Engine/`, this project implements a complete system to process files.
*   **Purpose**: A robust engine to transform unstructured text documents into validated JSON data.
*   **Key Components**:
    *   **`app.py`**: The CLI orchestration layer that handles user input, file ingestion, and LLM communication.
    *   **`core/extractor.py`**: Utility for safely fetching raw content from the local file system.
    *   **Data Samples**: Includes example inputs like `data/support_email_1.txt` and `data/resume_1.txt` for testing extraction logic.

## 🛠️ Key Technical Concepts
*   **Model-Level Validation**: Using Pydantic to ensure the relationship between fields is logically consistent.
*   **Error Handling**: Managing `ValidationError` exceptions to identify when an LLM fails to comply with the requested schema.
*   **Structured Extraction**: Enforcing schemas on LLM outputs to ensure data is ready for downstream programmatic use.

## 🚀 Getting Started
To explore the implementation of these concepts, refer to the individual `README.md` files within the `Project-AI-Document-Intelligence-Engine/` directory for setup instructions and workflow details.

---
*For further learning on Pydantic's advanced features, please refer to the official [Pydantic documentation](https://docs.pydantic.dev/).*
