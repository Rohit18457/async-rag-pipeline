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

### 1. Upload Document (Asynchronous PDF Ingestion)
Uploads and enqueues a PDF file for background parsing, chunking, and vector indexing into Qdrant.

- **URL:** `/documents/upload`
- **Method:** `POST`
- **Status:** `202 Accepted`
- **Content-Type:** `multipart/form-data`
- **Query Parameters:** `collection_name` *(optional, defaults to `learning_rag`)*

**cURL Example:**
```bash
curl -X POST "http://127.0.0.1:8000/documents/upload" \
     -F "file=@sample.pdf"
```

**Response:**
```json
{
  "status": "queued",
  "job_id": "f891a27e-3b56-4271-97b1-289569cdbfb1",
  "filename": "sample.pdf",
  "message": "Document 'sample.pdf' queued for asynchronous indexing. Track progress via /job-status/f891a27e-3b56-4271-97b1-289569cdbfb1"
}
```

**Track Ingestion Result (`GET /job-status/{job_id}`):**
```json
{
  "job_id": "f891a27e-3b56-4271-97b1-289569cdbfb1",
  "status": "finished",
  "result": {
    "status": "completed",
    "filename": "sample.pdf",
    "pages_processed": 25,
    "chunks_indexed": 82,
    "collection": "learning_rag"
  }
}
```

---

### 2. Submit Query
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

### 3. Poll Job Status / Retrieve Results
Fetches the processing state and output for any query or document ingestion job.

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

**Response (Finished Query — Structured with Citations & Scores):**
```json
{
  "job_id": "c3e98160-5a33-4f9e-97dc-3224f8d6899b",
  "status": "finished",
  "result": {
    "query": "What are the core modules in Node.js and their purposes?",
    "answer": "Based on the provided document (Page 12), Node.js includes several core modules such as `fs` for file system operations, `http` for creating web servers, and `path` for handling file paths...",
    "citations": [
      {
        "page": 12,
        "filename": "sample.pdf",
        "score": 0.8842,
        "snippet": "Node.js core modules include fs, http, https, path, and os. The fs module allows interacting with the file system..."
      },
      {
        "page": 14,
        "filename": "sample.pdf",
        "score": 0.8125,
        "snippet": "The http module provides facilities for HTTP transfer protocols and launching high-performance servers..."
      }
    ],
    "metrics": {
      "retrieval_ms": 28.45,
      "generation_ms": 312.18,
      "total_ms": 340.63
    }
  }
}
```

---

### 4. Health Check & Queue Depth
- **URL:** `/health`
- **Method:** `GET`

```json
{
  "status": "healthy",
  "redis_connected": true,
  "queue_depth": 0,
  "failed_jobs_count": 0
}
```

---

### 5. Dead Letter Queue (DLQ) Management
Jobs encountering transient network or API failures are automatically retried with **Exponential Backoff** (`interval=[5, 15, 30]` seconds). If all retries are exhausted, the job is quarantined in the Dead Letter Queue for inspection and manual requeueing.

#### Inspect Failed Jobs:
- **URL:** `/queue/failed-jobs`
- **Method:** `GET`

```json
{
  "failed_count": 1,
  "failed_jobs": [
    {
      "job_id": "a90184b2-c11f-4982-b7e4-2395a129ef31",
      "enqueued_at": "2026-11-22 11:45:10",
      "ended_at": "2026-11-22 11:46:02",
      "exc_info": "... GroqRateLimitError: Rate limit exceeded (429)"
    }
  ]
}
```

#### Requeue / Retry a Failed Job:
- **URL:** `/queue/failed-jobs/{job_id}/retry`
- **Method:** `POST`

```json
{
  "message": "Job 'a90184b2-c11f-4982-b7e4-2395a129ef31' successfully requeued from Dead Letter Queue to active queue.",
  "status": "queued"
}
```

---

### 6. Redis Exact-Match Query Cache
Identical questions bypass vector similarity search and Groq LLM inference completely, returning cached responses in **`< 2ms`** with zero token costs.

* **Cache Key:** Deterministic SHA-256 hash `rag_cache:{collection}:{hash}`.
* **Configurable TTL:** Defaults to 3600 seconds (`CACHE_TTL=3600`).
* **Auto-Invalidation:** When a new document is uploaded via `/documents/upload`, the cache for that collection is automatically flushed so answers reflect new data.

#### Inspect Cache Statistics:
- **URL:** `/cache/stats`
- **Method:** `GET`

```json
{
  "cached_queries_count": 14,
  "ttl_seconds": 3600,
  "sample_cache_keys": [
    "rag_cache:learning_rag:8f4b23...",
    "rag_cache:learning_rag:1a2c3d..."
  ]
}
```

#### Manually Flush Cache:
- **URL:** `/cache`
- **Method:** `DELETE`
- **Query Parameters:** `collection_name` *(optional, omit to flush all collections)*

```json
{
  "message": "Flushed all 14 cached queries across all collections.",
  "cleared_keys_count": 14
}
```

---

## 💡 Notes for Windows Users

RQ uses `os.fork()` by default, which is unsupported on Windows. This project provides `run_worker.py`, which utilizes `rq.SimpleWorker` to ensure smooth execution on Windows without errors.

---

## 📜 License

MIT License. Free to use and modify for educational and commercial projects.

