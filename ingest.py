"""
Document ingestion script for the RAG pipeline.
Extracts text from PDF documents, splits them into semantic chunks,
computes vector embeddings using Hugging Face, and stores them in Qdrant.
"""
import os
import sys
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()
logging.getLogger("pypdf").setLevel(logging.ERROR)

def ingest_pdf(pdf_path: str, qdrant_url: str = None, collection_name: str = None):
    qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
    collection_name = collection_name or os.getenv("COLLECTION_NAME", "learning_rag")
    embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

    path = Path(pdf_path)
    if not path.exists():
        print(f"❌ Error: File '{pdf_path}' not found.")
        sys.exit(1)

    print(f"📄 Loading document: {pdf_path}")
    loader = PyPDFLoader(file_path=str(path))
    docs = loader.load()
    print(f"   Loaded {len(docs)} page(s).")

    print("✂️  Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(docs)
    print(f"   Generated {len(chunks)} text chunks.")

    print(f"🧠 Initializing embedding model ({embedding_model_name})...")
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)

    print(f"💾 Indexing chunks into Qdrant collection '{collection_name}' at {qdrant_url}...")
    QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embedding_model,
        url=qdrant_url,
        collection_name=collection_name,
        force_recreate=True
    )
    print("✅ Ingestion complete! The collection is ready for queries.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a PDF document into Qdrant vector database.")
    parser.add_argument("pdf_path", nargs="?", default="sample.pdf", help="Path to PDF file to index (default: sample.pdf)")
    args = parser.parse_args()

    ingest_pdf(args.pdf_path)

