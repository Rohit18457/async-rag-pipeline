import os
import sys
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

load_dotenv()

from client.rq_client import redis_conn
from rq import SimpleWorker, Queue

if __name__ == "__main__":
    queue = Queue("default", connection=redis_conn)
    print("=" * 60)
    print("🚀 Starting RQ Worker (SimpleWorker for Windows / Local Dev)...")
    print("Listening on queue: 'default'")
    print("=" * 60)
    worker = SimpleWorker([queue], connection=redis_conn)
    worker.work()
