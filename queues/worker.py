import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from groq import Groq

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

def get_vector_db():
    global vector_db
    if vector_db is None:
        vector_db = QdrantVectorStore.from_existing_collection(
            embedding=embedding_model,
            url=QDRANT_URL,
            collection_name=COLLECTION_NAME
        )
    return vector_db

# Initialize Groq client
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found. Please set it in your .env file.")

client = Groq(api_key=api_key)


def process_query(query: str) -> str:
    """
    Worker task: retrieves relevant context from Qdrant vector database
    and synthesizes an answer using Groq LLM.
    """
    print(f"\n🔍 Processing query: '{query}'")
    
    try:
        db = get_vector_db()
        search_results = db.similarity_search(
            query=query,
            k=4
        )
    except Exception as e:
        error_msg = f"Failed to retrieve context from Qdrant: {e}"
        print(f"❌ {error_msg}")
        return error_msg

    if not search_results:
        return "I cannot find any relevant context in the provided documents to answer your question."

    context = "\n\n\n".join([
        f"Page Content: {result.page_content}\nPage Number: {result.metadata.get('page', 0) + 1}\nFile Location: {result.metadata.get('source', '')}"
        for result in search_results
    ])

    system_prompt = f"""
You are a helpful AI Assistant who answers user queries based strictly on the available context retrieved from a PDF file.
Always cite the relevant page number(s) in your answer when referencing information.
If the answer cannot be found in the provided context, politely state: "I cannot find this information in the provided document." Do not attempt to fabricate answers.

Context:
{context}
"""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
        )
        answer = response.choices[0].message.content
        print("✅ Query processed successfully.")
        return answer
    except Exception as e:
        error_msg = f"Error generating response from Groq: {e}"
        print(f"❌ {error_msg}")
        return error_msg
