"""Runtime configuration, read lazily so tests can change the environment."""
import os
from pathlib import Path


def load_env_file(path: str = ".env") -> None:
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env_file()

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_CHARS_PER_DOC = 600_000
MAX_DOCUMENTS = 60
MAX_TOTAL_CHUNKS = 6000


def gemini_key() -> str:
    return (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()


def gemini_model() -> str:
    return os.getenv("GEMINI_MODEL", "").strip()


def embed_model() -> str:
    return os.getenv("GEMINI_EMBED_MODEL", "").strip()


def database_path() -> str:
    explicit = os.getenv("DATABASE_PATH")
    if explicit:
        return explicit
    if os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"):
        return "/tmp/knowledge.db"  # serverless platforms only allow writing to /tmp
    return "knowledge.db"
