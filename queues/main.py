import os
import uvicorn
from dotenv import load_dotenv

try:
    from .server import app
except ImportError:
    from server import app

load_dotenv()

def main():
    host = os.getenv("APP_HOST", "127.0.0.1")
    port = int(os.getenv("APP_PORT", 8000))
    print(f"🚀 Starting API server on http://{host}:{port}")
    uvicorn.run(app, port=port, host=host)

if __name__ == "__main__":
    main()
