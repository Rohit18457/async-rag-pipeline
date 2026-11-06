import os
import sys
from dotenv import load_dotenv
from fastapi import FastAPI, Query, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional, Any

# Ensure RAG_Queue directory is in Python path for clean imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

load_dotenv()

from client.rq_client import queue, redis_conn
from queues.worker import process_query

app = FastAPI(
    title="Asynchronous RAG API",
    description="Scalable, distributed Retrieval-Augmented Generation API powered by FastAPI, Redis Queue (RQ), Qdrant Vector DB, and Groq LLMs.",
    version="1.0.0"
)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The question or prompt to answer using retrieved PDF context", example="What is asynchronous programming in Node.js?")


class ChatResponse(BaseModel):
    status: str
    job_id: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[Any] = None


@app.get('/', tags=["General"])
def root():
    return {
        "status": "online",
        "service": "Asynchronous RAG API",
        "docs_url": "/docs"
    }


@app.get('/health', tags=["General"])
def health_check():
    """Verify Redis connection health."""
    try:
        redis_alive = redis_conn.ping()
    except Exception:
        redis_alive = False

    return {
        "status": "healthy" if redis_alive else "degraded",
        "redis_connected": redis_alive
    }


@app.post('/chat', response_model=ChatResponse, status_code=status.HTTP_202_ACCEPTED, tags=["RAG Tasks"])
def submit_chat_job(body: ChatRequest):
    """
    Enqueue a query to be processed asynchronously by the background RQ worker.
    Returns immediately with a unique `job_id`.
    """
    try:
        job = queue.enqueue(process_query, body.query)
        return {"status": "queued", "job_id": job.id}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to enqueue task to Redis queue: {str(e)}"
        )


@app.get('/job-status/{job_id}', response_model=JobStatusResponse, tags=["RAG Tasks"])
def get_job_status_by_path(job_id: str):
    """
    Check the processing state and retrieve the answer for a specific `job_id`.
    Status can be: `queued`, `started`, `finished`, or `failed`.
    """
    job = queue.fetch_job(job_id=job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with ID '{job_id}' not found. It may have expired or was never created."
        )

    return {
        "job_id": job.id,
        "status": job.get_status(),
        "result": job.return_value()
    }


# Backward compatibility for query parameter style: /job-status?job_id=...
@app.get('/job-status', response_model=JobStatusResponse, tags=["RAG Tasks"], include_in_schema=False)
def get_job_status_by_query(job_id: str = Query(..., description="Job ID")):
    return get_job_status_by_path(job_id)
