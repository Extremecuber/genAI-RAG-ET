import os
from dotenv import load_dotenv

load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")

if not HF_API_TOKEN:
    raise RuntimeError("HF_API_TOKEN environment variable is not set")

HF_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

HF_ROUTER_BASE_URL = (
    "https://router.huggingface.co/hf-inference/models"
)

HF_HEADERS = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "application/json",
}
