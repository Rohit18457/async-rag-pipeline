import os
import sys
import uuid
import shutil
from dotenv import load_dotenv
from fastapi import FastAPI, Query, HTTPException, status, File, UploadFile
from pydantic import BaseModel, Field
from typing import Optional, Any, List

# Ensure RAG_Queue directory is in Python path for clean imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

load_dotenv()

from client.rq_client import queue, redis_conn, failed_registry, Retry
from queues.worker import process_query, ingest_document_task, invalidate_collection_cache

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(
    title="Asynchronous RAG API",
    description="Scalable, distributed Retrieval-Augmented Generation API powered by FastAPI, Redis Queue (RQ) with Exponential Backoff Retries, Redis Exact-Match Query Cache, Qdrant Vector DB, and Groq LLMs.",
    version="1.3.0"
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


class Citation(BaseModel):
    page: int = Field(..., description="Document page number")
    filename: str = Field(..., description="Source document file name")
    score: float = Field(..., description="Semantic cosine similarity score")
    snippet: str = Field(..., description="Extracted context snippet from the page")


class QueryMetrics(BaseModel):
    retrieval_ms: float = Field(..., description="Vector similarity search duration in milliseconds")
    generation_ms: float = Field(..., description="LLM inference synthesis duration in milliseconds")
    total_ms: float = Field(..., description="Total processing time in milliseconds")
    cache_hit: Optional[bool] = Field(False, description="Whether the query was served from the Redis cache")


class QueryResult(BaseModel):
    query: str
    answer: str
    citations: List[Citation]
    cached: bool = Field(False, description="Whether this response was served from the Redis Exact-Match Cache (<2ms)")
    metrics: QueryMetrics


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    retries_left: Optional[int] = Field(None, description="Remaining automatic retries if job encountered a transient error")
    result: Optional[Any] = Field(None, description="Result object (QueryResult for chat, ingestion metadata for uploads)")


class FailedJobItem(BaseModel):
    job_id: str
    enqueued_at: Optional[str] = None
    ended_at: Optional[str] = None
    exc_info: Optional[str] = None


class FailedJobsResponse(BaseModel):
    failed_count: int
    failed_jobs: List[FailedJobItem]


class CacheStatsResponse(BaseModel):
    cached_queries_count: int
    ttl_seconds: int
    sample_cache_keys: List[str]


# =============================================================================
# General & Health Endpoints
# =============================================================================

@app.get('/', tags=["General"])
def root():
    return {
        "status": "online",
        "service": "Asynchronous RAG API",
        "version": "1.3.0",
        "features": [
            "Async Query Processing",
            "Async PDF Ingestion",
            "Structured Response with Citations & Scores",
            "Auto-Retries with Exponential Backoff",
            "Dead Letter Queue (DLQ)",
            "Redis Exact-Match Query Cache (<2ms hits)"
        ],
        "docs_url": "/docs"
    }


@app.get('/health', tags=["General"])
def health_check():
    """Verify Redis connection, queue depth, and cache health."""
    try:
        redis_alive = redis_conn.ping()
        cache_count = len(list(redis_conn.scan_iter(match="rag_cache:*", count=100)))
    except Exception:
        redis_alive = False
        cache_count = 0

    return {
        "status": "healthy" if redis_alive else "degraded",
        "redis_connected": redis_alive,
        "queue_depth": len(queue),
        "failed_jobs_count": len(failed_registry),
        "cached_queries_count": cache_count
    }


# =============================================================================
# Asynchronous Query Processing (with Exponential Backoff Retries)
# =============================================================================

@app.post('/chat', response_model=ChatResponse, status_code=status.HTTP_202_ACCEPTED, tags=["RAG Tasks"])
def submit_chat_job(body: ChatRequest):
    """
    Enqueue a query to be processed asynchronously by the background RQ worker.
    Configured with Automatic Exponential Backoff Retries (3 retries with 5s, 15s, 30s delays)
    and Redis Exact-Match caching.
    """
    try:
        job = queue.enqueue(
            process_query,
            body.query,
            body.collection_name,
            retry=Retry(max=3, interval=[5, 15, 30])
        )
        return {"status": "queued", "job_id": job.id}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to enqueue task to Redis queue: {str(e)}"
        )


# =============================================================================
# Asynchronous Document Ingestion (with Auto-Retries & Cache Invalidation)
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
    - Dispatches a background ingestion job to the Redis Queue (RQ) with automatic retries.
    - Invalidates query cache on success so answers stay fresh.
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

    # Enqueue background ingestion task with generous timeout & 2 retries (10s, 30s)
    try:
        job = queue.enqueue(
            ingest_document_task,
            saved_path,
            file.filename,
            collection_name,
            job_timeout=600,
            retry=Retry(max=2, interval=[10, 30])
        )
        return {
            "status": "queued",
            "job_id": job.id,
            "filename": file.filename,
            "message": f"Document '{file.filename}' queued for asynchronous indexing with auto-retry. Track progress via /job-status/{job.id}"
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
    Shows remaining retries if currently recovering from transient errors.
    """
    job = queue.fetch_job(job_id=job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with ID '{job_id}' not found. It may have expired or was never created."
        )

    retries_left = job.retries_left if hasattr(job, "retries_left") else None

    return {
        "job_id": job.id,
        "status": job.get_status(),
        "retries_left": retries_left,
        "result": job.return_value()
    }


# Backward compatibility for query parameter style: /job-status?job_id=...
@app.get('/job-status', response_model=JobStatusResponse, tags=["RAG Tasks"], include_in_schema=False)
def get_job_status_by_query(job_id: str = Query(..., description="Job ID")):
    return get_job_status_by_path(job_id)


# =============================================================================
# Dead Letter Queue (DLQ) & Failure Recovery
# =============================================================================

@app.get('/queue/failed-jobs', response_model=FailedJobsResponse, tags=["Queue Management"])
def list_failed_jobs():
    """
    Inspect jobs that permanently failed after all exponential retries (Dead Letter Queue).
    """
    job_ids = failed_registry.get_job_ids()
    failed_items = []

    for j_id in job_ids:
        job = queue.fetch_job(j_id)
        if job:
            failed_items.append(
                FailedJobItem(
                    job_id=job.id,
                    enqueued_at=str(job.enqueued_at) if job.enqueued_at else None,
                    ended_at=str(job.ended_at) if job.ended_at else None,
                    exc_info=(job.exc_info or "")[-400:]
                )
            )

    return {
        "failed_count": len(failed_items),
        "failed_jobs": failed_items
    }


@app.post('/queue/failed-jobs/{job_id}/retry', tags=["Queue Management"])
def retry_failed_job(job_id: str):
    """
    Manually requeue a failed job from the Dead Letter Queue back into the active processing queue.
    """
    job = queue.fetch_job(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job '{job_id}' not found.")

    failed_registry.requeue(job_id)
    return {
        "message": f"Job '{job_id}' successfully requeued from Dead Letter Queue to active queue.",
        "status": "queued"
    }


# =============================================================================
# Redis Exact-Match Query Cache Management
# =============================================================================

@app.get('/cache/stats', response_model=CacheStatsResponse, tags=["Cache Management"])
def get_cache_stats():
    """Inspect Redis Exact-Match Query Cache statistics, active keys, and configured TTL."""
    try:
        keys = [k.decode('utf-8') if isinstance(k, bytes) else str(k) for k in redis_conn.scan_iter(match="rag_cache:*", count=100)]
    except Exception:
        keys = []

    ttl_val = int(os.getenv("CACHE_TTL", 3600))
    return {
        "cached_queries_count": len(keys),
        "ttl_seconds": ttl_val,
        "sample_cache_keys": keys[:10]
    }


@app.delete('/cache', tags=["Cache Management"])
def flush_query_cache(collection_name: Optional[str] = Query(None, description="Optional collection name to flush (omit to flush all collections)")):
    """Flush cached queries from Redis."""
    if collection_name:
        count = invalidate_collection_cache(collection_name)
        msg = f"Flushed {count} cached queries for collection '{collection_name}'."
    else:
        try:
            keys = list(redis_conn.scan_iter(match="rag_cache:*", count=100))
            count = redis_conn.delete(*keys) if keys else 0
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to flush cache: {str(e)}")
        msg = f"Flushed all {count} cached queries across all collections."

    return {"message": msg, "cleared_keys_count": count}
