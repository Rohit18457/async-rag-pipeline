import os
import sys
import uuid
import shutil
from dotenv import load_dotenv
from fastapi import FastAPI, Query, HTTPException, status, File, UploadFile
from pydantic import BaseModel, Field
from typing import Optional, Any

# Ensure RAG_Queue directory is in Python path for clean imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

load_dotenv()

from client.rq_client import queue, redis_conn
from queues.worker import process_query, ingest_document_task

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(
    title="Asynchronous RAG API",
    description="Scalable, distributed Retrieval-Augmented Generation API powered by FastAPI, Redis Queue (RQ), Qdrant Vector DB, and Groq LLMs.",
    version="1.1.0"
)


# =============================================================================
# Request & Response Schemas
# =============================================================================

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The question or prompt to answer using retrieved PDF context", example="What is asynchronous programming in Node.js?")
    collection_name: Optional[str] = Field(None, description="Optional target collection to query")


class ChatResponse(BaseModel):
    status: str
    job_id: str


class UploadResponse(BaseModel):
    status: str
    job_id: str
    filename: str
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[Any] = None


# =============================================================================
# Endpoints
# =============================================================================

@app.get('/', tags=["General"])
def root():
    return {
        "status": "online",
        "service": "Asynchronous RAG API",
        "version": "1.1.0",
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


# =============================================================================
# Asynchronous Query Processing
# =============================================================================

@app.post('/chat', response_model=ChatResponse, status_code=status.HTTP_202_ACCEPTED, tags=["RAG Tasks"])
def submit_chat_job(body: ChatRequest):
    """
    Enqueue a query to be processed asynchronously by the background RQ worker.
    Returns immediately with a unique `job_id`.
    """
    try:
        job = queue.enqueue(process_query, body.query, body.collection_name)
        return {"status": "queued", "job_id": job.id}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to enqueue task to Redis queue: {str(e)}"
        )


# =============================================================================
# Asynchronous Document Ingestion
# =============================================================================

@app.post(
    '/documents/upload',
    response_model=UploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Document Ingestion"]
)
def upload_document(
    file: UploadFile = File(..., description="PDF document to parse, chunk, and index into vector storage"),
    collection_name: Optional[str] = Query(None, description="Optional target Qdrant collection name (default: learning_rag)")
):
    """
    Asynchronously upload and ingest a PDF document.
    - Saves uploaded file to disk.
    - Dispatches a background ingestion job to the Redis Queue (RQ).
    - Returns immediately with HTTP 202 and a `job_id`.
    - Check progress/results at `/job-status/{job_id}`.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Only PDF files are supported."
        )

    # Generate unique temporary file name to prevent collision
    unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
    saved_path = os.path.join(UPLOAD_DIR, unique_filename)

    try:
        with open(saved_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded file: {str(e)}"
        )

    # Enqueue background ingestion task with generous timeout (10 mins)
    try:
        job = queue.enqueue(
            ingest_document_task,
            saved_path,
            file.filename,
            collection_name,
            job_timeout=600
        )
        return {
            "status": "queued",
            "job_id": job.id,
            "filename": file.filename,
            "message": f"Document '{file.filename}' queued for asynchronous indexing. Track progress via /job-status/{job.id}"
        }
    except Exception as e:
        if os.path.exists(saved_path):
            os.remove(saved_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to enqueue ingestion task to Redis queue: {str(e)}"
        )


# =============================================================================
# Job Status Tracking
# =============================================================================

@app.get('/job-status/{job_id}', response_model=JobStatusResponse, tags=["RAG Tasks"])
def get_job_status_by_path(job_id: str):
    """
    Check the processing state and retrieve results for a specific `job_id`.
    Works for both query jobs (`/chat`) and ingestion jobs (`/documents/upload`).
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
