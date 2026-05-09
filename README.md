# The RAG Challenge

## Introduction

This project explores Retrieval-Augmented Generation (RAG) using an extended knowledge base inspired by a fictional insurance company, InsureLLM.

## Project Structure

**Root directory:**

- `uv run ingest.py` (from `implementation/`): Ingest the latest knowledge base data.
- `uv run app.py`: Run the Q&A Chatbot for interactive conversations.
- `uv run evaluator.py`: Evaluate chatbot performance using the available evaluation scripts.

**implementation directory:**

- `answer.py`: Handles question answering. Customize or enhance `fetch_context()` and `answer_question()` as you see fit!
- `ingest.py`: Responsible for ingestion and preprocessing of the knowledge base. This is fully open to your improvements and ideas.

**evaluation directory:**

- Contains scripts for automated evaluation on test data (do not modify for consistency in evaluation).

## Objective

- Understand the project structure and current implementation.
- Re-implement or improve on `ingest.py` and `answer.py` with your approaches.
- Strive for the most effective, high-performing Q&A chatbot!

Good luck!

