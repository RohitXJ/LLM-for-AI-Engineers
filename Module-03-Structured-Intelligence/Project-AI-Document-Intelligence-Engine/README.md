# Project: AI Document Intelligence Engine

This project is a component of the **Module 03: Structured Logic & Extraction** curriculum. It provides a robust, production-grade engine to transform unstructured text documents into validated, structured JSON data using Large Language Models.

## 🏗️ Architectural Overview
The engine leverages the **Instructor** library to enforce **Pydantic** schemas on LLM outputs, ensuring that extracted data is reliable, type-safe, and ready for downstream integration.

### Core Components
*   **`app_streamlit.py`**: The orchestration layer. Uses Streamlit to provide an intuitive UI, manage session state for the LLM Manager, and handle concurrent file processing.
*   **`core/models.py`**: Contains the "Source of Truth" for data structure. Uses Pydantic models to define the schemas for Resumes, Invoices, and Support Emails.
*   **`core/llm.py`**: Manages LLM connectivity. Implements a `Manager` class that abstracts Ollama interaction, handling synchronous tasks via a `ThreadPoolExecutor` and exposing an asynchronous API to the UI.
*   **`core/templates.py`**: Houses the system prompts. These define the "persona" and extraction logic for each document type, ensuring the LLM acts as a specialized analyzer for each format.
*   **`core/extractor.py`**: Provides basic file system utilities to fetch raw content from text/markdown files.

## 🚀 Key Features
*   **Structured Extraction**: Uses Instructor + Pydantic to guarantee consistent JSON output.
*   **Concurrency**: Asynchronous processing allows the engine to handle multiple documents in parallel, drastically reducing wait times.
*   **Schema-Driven Design**: Adding support for new document types is as simple as defining a new Pydantic model and a corresponding prompt template.
*   **Persistence**: Automatically saves all successful extractions to JSON files in the `out/` directory.

## 🛠️ Getting Started

### Prerequisites
*   [Ollama](https://ollama.ai/) running locally.
*   `pip install streamlit instructor openai pydantic`

### Running the Application
1.  Ensure Ollama is running:
    ```bash
    ollama serve
    ```
2.  Launch the application:
    ```bash
    streamlit run app_streamlit.py
    ```

### Workflow
1.  **Select Document Type**: Choose between Resume, Invoice, or Email from the sidebar.
2.  **Upload**: Select one or more `.txt` or `.md` files.
3.  **Process**: Click "Process Documents". The app will handle the extraction, update the progress bar, and display the structured results.
4.  **Export**: All processed JSON files can be found in the `out/` directory.
