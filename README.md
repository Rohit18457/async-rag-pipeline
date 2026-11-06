# ⚡ Asynchronous Distributed RAG Pipeline

A production-ready, asynchronous **Retrieval-Augmented Generation (RAG)** system built with **FastAPI**, **Redis Queue (RQ)**, **Qdrant Vector Database**, and **Groq Cloud LLMs**.

---

## 🌟 Overview

Standard RAG APIs often suffer from latency: retrieving dense embeddings and waiting for an LLM response blocks the HTTP request worker, quickly exhausting connection pools under concurrent load.

This project solves this bottleneck by implementing a **fully decoupled, asynchronous job queue architecture**:
1. **FastAPI** accepts incoming queries and immediately returns a job ticket (`job_id`) with HTTP `202 Accepted`.
2. **Valkey / Redis** queues the query task.
3. **RQ Background Worker** consumes the task, performs semantic similarity search against **Qdrant**, constructs the augmented prompt, and queries the **Groq LLM**.
4. The client polls the status endpoint or retrieves the answer once finished.

---

## 🏗️ Architecture

```
                       +-------------------+
                       |    Client / UI    |
                       +---------+---------+
                                 |
           1. POST /chat         |       4. GET /job-status/{job_id}
      (Returns { "job_id" })     |          (Returns final result)
                                 v
                       +-------------------+
                       |    FastAPI App    |
                       +---------+---------+
                                 |
                                 | 2. Enqueue task
                                 v
                       +-------------------+
                       |   Valkey / Redis  |
                       |    (Port 6379)    |
                       +---------+---------+
                                 |
                                 | 3. Dequeue & execute
                                 v
                       +-------------------+
                       |   RQ Worker Pool  |
                       +----+---------+----+
                            |         |
     3a. Similarity Search  |         | 3b. Synthesize Answer
                            v         v
                     +------------+ +------------+
                     |   Qdrant   | |  Groq LLM  |
                     |  (Port     | | (Qwen/Llama|
                     |   6333)    | |  Inference)|
                     +------------+ +------------+
```

---

## 🛠️ Tech Stack

- **Framework:** [FastAPI](https://fastapi.tiangolo.com/) with [Pydantic v2](https://docs.pydantic.dev/)
- **Task Queue:** [Redis Queue (RQ)](https://python-rq.org/) + [Valkey / Redis](https://valkey.io/)
- **Vector Database:** [Qdrant](https://qdrant.tech/)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` via [LangChain HuggingFace](https://python.langchain.com/)
- **LLM Provider:** [Groq](https://groq.com/) (Qwen 3.8 / Llama 3)
- **Containerization:** Docker & Docker Compose

---

## 📁 Repository Structure

```
RAG_Queue/
├── client/
│   ├── __init__.py
│   └── rq_client.py         # Redis and RQ queue connection setup
├── queues/
│   ├── __init__.py
│   ├── main.py              # Server entry point (Uvicorn)
│   ├── server.py            # FastAPI REST endpoints & Pydantic models
│   └── worker.py            # Background worker task (Qdrant + Groq)
├── docker-compose.yml       # Qdrant and Valkey multi-container setup
├── ingest.py                # CLI script to chunk & embed PDFs into Qdrant
├── run_worker.py            # Cross-platform RQ worker runner (Windows-safe)
├── sample.pdf               # Sample document for quick testing
├── requirements.txt         # Pinned Python dependencies
├── .env.example             # Template for required environment variables
└── README.md
```

---

## 🚀 Getting Started

### 1. Prerequisites
- [Python 3.10+](https://www.python.org/)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- A free [Groq Cloud API Key](https://console.groq.com/keys)

---

### 2. Clone the Repository
```bash
git clone https://github.com/Rohit18457/async-rag-pipeline.git
cd async-rag-pipeline
```

---

### 3. Create and Activate Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

---

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

### 5. Configure Environment Variables
Copy `.env.example` to `.env` and fill in your Groq API key:
```bash
cp .env.example .env
```

Edit `.env`:
```env
GROQ_API_KEY=gsk_your_actual_key_here
GROQ_MODEL=qwen/qwen3.8-27b
REDIS_HOST=localhost
REDIS_PORT=6379
QDRANT_URL=http://localhost:6333
COLLECTION_NAME=learning_rag
```

---

### 6. Start the Database Containers
Start Qdrant and Valkey using Docker Compose:
```bash
docker compose up -d
```
Verify they are running:
- Qdrant Web UI: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)
- Valkey (Redis): Port `6379`

---

### 7. Index a Document into Qdrant
Run the ingestion script to process and index the sample PDF:
```bash
python ingest.py sample.pdf
```
*(You can pass any custom PDF path: `python ingest.py path/to/your_file.pdf`)*

---

### 8. Run the Application

Open two separate terminal windows:

#### Terminal 1 — Start the API Server:
```bash
python queues/main.py
```
Server runs at `http://127.0.0.1:8000`. Interactive Swagger documentation is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

#### Terminal 2 — Start the Background Worker:
```bash
python run_worker.py
```
*(On Linux/macOS, you can also use `rq worker`)*

---

## 📡 API Reference

### 1. Submit Query
Enqueues a user query for asynchronous processing.

- **URL:** `/chat`
- **Method:** `POST`
- **Status:** `202 Accepted`

**Request Body:**
```json
{
  "query": "What are the core modules in Node.js and their purposes?"
}
```

**Response:**
```json
{
  "status": "queued",
  "job_id": "c3e98160-5a33-4f9e-97dc-3224f8d6899b"
}
```

---

### 2. Poll Job Status / Retrieve Answer
Fetches the processing state and final answer.

- **URL:** `/job-status/{job_id}`
- **Method:** `GET`

**Response (While Processing):**
```json
{
  "job_id": "c3e98160-5a33-4f9e-97dc-3224f8d6899b",
  "status": "started",
  "result": null
}
```

**Response (Finished):**
```json
{
  "job_id": "c3e98160-5a33-4f9e-97dc-3224f8d6899b",
  "status": "finished",
  "result": "Based on the provided document (Page 12), Node.js includes several core modules..."
}
```

---

### 3. Health Check
- **URL:** `/health`
- **Method:** `GET`
```json
{
  "status": "healthy",
  "redis_connected": true
}
```

---

## 💡 Notes for Windows Users

RQ uses `os.fork()` by default, which is unsupported on Windows. This project provides `run_worker.py`, which utilizes `rq.SimpleWorker` to ensure smooth execution on Windows without errors.

---

## 📜 License

MIT License. Free to use and modify for educational and commercial projects.

