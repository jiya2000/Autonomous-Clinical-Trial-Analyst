# Autonomous Clinical Trial Analyst

An end-to-end clinical trial question-answering app that runs fully locally.
It uses a small local LLM (via Ollama), a vector database (Qdrant), and a
LangGraph agent workflow to analyse and interrogate clinical trial protocols
from ClinicalTrials.gov.

## Features

- Chat UI to ask natural language questions about clinical trials
- Local LLM only (no paid APIs) using TinyLlama via Ollama
- Retrieval-Augmented Generation over a curated set of diabetes trials
- Vector search powered by Qdrant
- Multi-step agent workflow (Supervisor → Retrieval → Analysis) with
  an "Agent Thought Trace" panel for transparency
- Docker-based deployment (backend + frontend + Qdrant)

## Architecture

- **Frontend:** React + TypeScript single-page app (served on port 3000)
- **Backend:** FastAPI + LangGraph orchestrating the agent workflow
- **Vector DB:** Qdrant collection `clinical_protocols`
- **LLM:** TinyLlama served locally via Ollama using the OpenAI-compatible API
- **Data:** Diabetes-related trials fetched from ClinicalTrials.gov v2 API and
  ingested into Qdrant

## Prerequisites

- Docker and Docker Compose installed
- Python 3.11+ (for running ingestion scripts locally)
- [Ollama](https://ollama.com/) installed with TinyLlama pulled:

  ```bash
  ollama pull tinyllama
  ollama serve
  ```

## Setup & Run

From the project root:

1. **Start the LLM locally** (once per machine):

   ```bash
   ollama pull tinyllama   # if not already pulled
   ollama serve
   ```

2. **(Optional) Create and activate a Python venv for backend tools:**

   ```bash
   cd backend
   python -m venv .venv
   .venv\Scripts\activate    # PowerShell on Windows
   pip install -r requirements.txt
   ```

3. **Fetch diabetes trials from ClinicalTrials.gov (v2 API):**

   ```bash
   # from project root
   cd backend
   python -m app.data_get.dataget
   ```

   This creates `diabetes_trials.json` at the project root.

4. **Ingest trials into Qdrant:**

   Make sure Qdrant is running (via Docker, see next step), then from `backend`:

   ```bash
   set QDRANT_URL=http://localhost:6333   # Windows cmd
   # or: $env:QDRANT_URL = "http://localhost:6333"  # PowerShell

   .venv\Scripts\python.exe -m app.ingestion.ingest_diabetes_trials
   ```

5. **Start the full stack with Docker:**

   From the project root:

   ```bash
   docker compose up --build
   ```

   This starts:
   - Frontend on `http://localhost:3000`
   - Backend FastAPI on `http://localhost:8000`
   - Qdrant on `http://localhost:6333`

6. **Use the app:**

   Open `http://localhost:3000` in your browser and:

   - Ask questions like:
     - "Summarize the main goals of one type 2 diabetes trial."
     - "What are typical inclusion and exclusion criteria across these diabetes trials?"
   - Watch the right-hand *Agent Thought Trace* panel to see how the
     Supervisor, Retrieval Agent, and Analysis Agent reason about each query.

## Notes

- The app currently focuses on a subset of diabetes-related trials; you can
  modify `app/data_get/dataget.py` and `app/ingestion/ingest_diabetes_trials.py`
  to ingest other conditions.
- All model calls are local via Ollama; you can swap TinyLlama for another
  compatible model by changing the `model` name in `backend/app/agents.py`.
