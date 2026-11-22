import os
import time
import logging
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from groq import Groq

logging.getLogger("pypdf").setLevel(logging.ERROR)
load_dotenv()

# Configuration from environment variables
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "learning_rag")
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# Initialize local embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME
)

vector_db = None

def get_vector_db(collection_name: str = None):
    global vector_db
    target_collection = collection_name or COLLECTION_NAME
    if vector_db is None or vector_db.collection_name != target_collection:
        vector_db = QdrantVectorStore.from_existing_collection(
            embedding=embedding_model,
            url=QDRANT_URL,
            collection_name=target_collection
        )
    return vector_db

# Initialize Groq client
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found. Please set it in your .env file.")

client = Groq(api_key=api_key)


# =============================================================================
# 1. Background Query Synthesis Task (Structured Response + Citations & Scores)
# =============================================================================

def process_query(query: str, collection_name: str = None) -> dict:
    """
    Worker task: retrieves relevant context with similarity scores from Qdrant,
    synthesizes an answer using Groq LLM, and returns structured output
    including citations, similarity scores, and execution latency metrics.
    """
    print(f"\n🔍 [Worker Query] Processing: '{query}'")

    # 1. Measure Retrieval Latency & Similarity Search with Scores
    start_retrieval = time.perf_counter()
    try:
        db = get_vector_db(collection_name=collection_name)
        search_results = db.similarity_search_with_score(
            query=query,
            k=4
        )
        retrieval_ms = (time.perf_counter() - start_retrieval) * 1000
    except Exception as e:
        error_msg = f"Failed to retrieve context from Qdrant: {e}"
        print(f"❌ {error_msg}")
        return {
            "query": query,
            "answer": error_msg,
            "citations": [],
            "status": "error"
        }

    if not search_results:
        return {
            "query": query,
            "answer": "I cannot find any relevant context in the provided documents to answer your question.",
            "citations": [],
            "metrics": {
                "retrieval_ms": round(retrieval_ms, 2),
                "generation_ms": 0.0,
                "total_ms": round(retrieval_ms, 2)
            }
        }

    # 2. Extract structured citations with similarity scores
    citations = []
    context_blocks = []

    for doc, score in search_results:
        page_num = doc.metadata.get("page", 0) + 1
        source_name = doc.metadata.get("filename") or os.path.basename(doc.metadata.get("source", "document.pdf"))
        snippet = doc.page_content.strip()

        citations.append({
            "page": page_num,
            "filename": source_name,
            "score": round(float(score), 4),
            "snippet": snippet[:200] + ("..." if len(snippet) > 200 else "")
        })

        context_blocks.append(
            f"[Source: {source_name} | Page: {page_num} | Relevance Score: {round(float(score), 4)}]\n{snippet}"
        )

    context = "\n\n---\n\n".join(context_blocks)

    system_prompt = f"""
You are a helpful AI Assistant who answers user queries based strictly on the available context retrieved from a PDF file.
Always cite the relevant page number(s) in your answer when referencing information.
If the answer cannot be found in the provided context, politely state: "I cannot find this information in the provided document." Do not attempt to fabricate answers.

Context:
{context}
"""

    # 3. Measure Generation Latency & Synthesize Answer
    start_gen = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
        )
        generation_ms = (time.perf_counter() - start_gen) * 1000
        answer = response.choices[0].message.content
        total_ms = retrieval_ms + generation_ms

        print(f"✅ Query completed in {total_ms:.1f}ms (Retrieval: {retrieval_ms:.1f}ms, Generation: {generation_ms:.1f}ms).")

        return {
            "query": query,
            "answer": answer,
            "citations": citations,
            "metrics": {
                "retrieval_ms": round(retrieval_ms, 2),
                "generation_ms": round(generation_ms, 2),
                "total_ms": round(total_ms, 2)
            }
        }

    except Exception as e:
        error_msg = f"Error generating response from Groq: {e}"
        print(f"❌ {error_msg}")
        return {
            "query": query,
            "answer": error_msg,
            "citations": citations,
            "status": "error"
        }


# =============================================================================
# 2. Background Document Ingestion Task (Asynchronous PDF Ingestion)
# =============================================================================

def ingest_document_task(file_path: str, original_filename: str, collection_name: str = None) -> dict:
    """
    Background worker task:
    1. Loads a PDF document from file_path.
    2. Splits it into semantic chunks with overlap.
    3. Computes vector embeddings using Hugging Face.
    4. Indexes chunks into Qdrant collection (creating if needed or appending).
    5. Safely cleans up temporary upload file.
    Returns structured metadata on the indexed document.
    """
    target_collection = collection_name or COLLECTION_NAME
    path = Path(file_path)

    print(f"\n📥 [Worker Ingestion] Processing document: '{original_filename}'")

    if not path.exists():
        error_msg = f"File '{file_path}' not found on worker."
        print(f"❌ {error_msg}")
        return {"status": "failed", "error": error_msg, "filename": original_filename}

    try:
        # 1. Load PDF pages
        loader = PyPDFLoader(file_path=str(path))
        docs = loader.load()
        num_pages = len(docs)
        print(f"   Loaded {num_pages} page(s) from '{original_filename}'.")

        if not docs:
            return {
                "status": "failed",
                "error": "Document contains no readable text or pages.",
                "filename": original_filename
            }

        # Enrich chunk metadata
        for doc in docs:
            doc.metadata["filename"] = original_filename

        # 2. Split document into semantic chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(docs)
        num_chunks = len(chunks)
        print(f"   Generated {num_chunks} text chunks.")

        # 3. Index into Qdrant Vector DB
        try:
            store = QdrantVectorStore.from_existing_collection(
                embedding=embedding_model,
                url=QDRANT_URL,
                collection_name=target_collection
            )
            store.add_documents(chunks)
        except Exception:
            # Collection does not exist yet; create it
            QdrantVectorStore.from_documents(
                documents=chunks,
                embedding=embedding_model,
                url=QDRANT_URL,
                collection_name=target_collection
            )

        print(f"✅ Ingestion successful: '{original_filename}' indexed into collection '{target_collection}'.")

        # 4. Cleanup temporary uploaded file
        try:
            if path.exists():
                os.remove(path)
        except Exception:
            pass

        return {
            "status": "completed",
            "filename": original_filename,
            "pages_processed": num_pages,
            "chunks_indexed": num_chunks,
            "collection": target_collection
        }

    except Exception as e:
        error_msg = f"Failed to ingest document '{original_filename}': {str(e)}"
        print(f"❌ {error_msg}")
        try:
            if path.exists():
                os.remove(path)
        except Exception:
            pass
        return {
            "status": "failed",
            "error": error_msg,
            "filename": original_filename
        }
