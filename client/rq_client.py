import os
from dotenv import load_dotenv
from redis import Redis
from rq import Queue, Retry
from rq.registry import FailedJobRegistry

load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

redis_conn = Redis(host=REDIS_HOST, port=REDIS_PORT)
queue = Queue("default", connection=redis_conn)
failed_registry = FailedJobRegistry(queue=queue)
