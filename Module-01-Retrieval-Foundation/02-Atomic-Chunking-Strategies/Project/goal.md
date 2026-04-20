# Project: The Multi-Strategy Chunking CLI 🛠️

## 🎯 Goal
Build a production-ready Command Line Interface (CLI) tool that transforms raw text files into structured, "Vector-DB-Ready" chunks using multiple chunking strategies.

## 🚀 Objectives
1.  **Strategy Selection**: Implement at least three chunking methods:
    *   `RecursiveCharacter` (The Workhorse)
    *   `Token` (The Cost-Optimizer)
    *   `Semantic` (The Context-Preserver)
2.  **Metadata Injection**: Every chunk must carry its own "passport" (metadata) including source file name, chunk index, and strategy used.
3.  **Persistence**: Save the output into a structured `.json` file that can be directly imported into a database like ChromaDB or Qdrant.
4.  **Validation**: Include a summary report at the end showing total chunks created and average characters per chunk.

## 📥 Input Requirements
*   **Source**: A `.txt` file containing long-form content (e.g., a technical article or a chat log).
*   **Parameters**:
    *   `chunk_size`: Targeted size of each piece.
    *   `chunk_overlap`: How much context to carry over.
    *   `strategy`: Choice of algorithm.

## 📤 Output Requirements
*   A `processed_chunks.json` file containing an array of objects.
*   A terminal summary showing the "Efficiency Score" of the operation.
