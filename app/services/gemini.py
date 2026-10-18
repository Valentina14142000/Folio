"""Minimal Gemini REST client (standard library): text generation and embeddings."""
import json
import math
import time
import urllib.error
import urllib.request
from array import array
from typing import Any, Optional

from app import config

BASE = "https://generativelanguage.googleapis.com/v1beta"
EMBED_DIMS = 768


class AIUnavailable(Exception):
    """The AI provider cannot be used right now; callers fall back to non-AI behaviour."""


def _post(url: str, key: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"content-type": "application/json", "x-goog-api-key": key},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _call(models: list[str], path_suffix: str, body_for: Any, budget: float) -> tuple[str, dict[str, Any]]:
    """Try models in order with light retries. Returns (model, payload)."""
    key = config.gemini_key()
    if not key:
        raise AIUnavailable("no API key configured")
    deadline = time.monotonic() + budget
    last = "unknown error"
    for model in dict.fromkeys(m for m in models if m):
        for attempt in range(2):
            left = deadline - time.monotonic()
            if left < 2:
                raise AIUnavailable("timed out")
            try:
                payload = _post(f"{BASE}/models/{model}:{path_suffix}", key, body_for(model), min(25, left))
                return model, payload
            except urllib.error.HTTPError as e:
                if e.code in (401, 403):
                    raise AIUnavailable("API key rejected") from e
                if e.code == 429:
                    raise AIUnavailable("rate limited") from e
                last = f"HTTP {e.code}"
                if e.code in (400, 404):
                    break  # model not available for this call: try the next one
            except (urllib.error.URLError, TimeoutError, OSError, ValueError):
                last = "network error"
            time.sleep(0.4 * (attempt + 1))
    raise AIUnavailable(last)


# ------------------------------------------------------------------ embeddings
def normalize(vec: list[float]) -> array:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return array("f", (v / norm for v in vec))


def embed_texts(texts: list[str], task: str) -> tuple[str, list[array]]:
    """Embed texts (task: RETRIEVAL_DOCUMENT or RETRIEVAL_QUERY). Returns (model, unit vectors)."""
    models = [config.embed_model(), "gemini-embedding-001", "text-embedding-004"]
    out: list[array] = []
    used = ""
    for start in range(0, len(texts), 50):
        batch = texts[start:start + 50]

        def body(model: str, batch=batch) -> dict[str, Any]:
            return {"requests": [{
                "model": f"models/{model}",
                "content": {"parts": [{"text": t[:6000]}]},
                "taskType": task,
                "outputDimensionality": EMBED_DIMS,
            } for t in batch]}

        # keep the same model for every batch of one call, so vectors are comparable
        model, payload = _call([used] if used else models, "batchEmbedContents", body, 40)
        used = model
        vectors = [e["values"] for e in payload.get("embeddings", [])]
        if len(vectors) != len(batch):
            raise AIUnavailable("unexpected embedding response")
        out.extend(normalize(v) for v in vectors)
    return used, out


# ------------------------------------------------------------------ generation
def generate_text(prompt: str, temperature: float = 0.2) -> str:
    models = [config.gemini_model(), "gemini-flash-latest", "gemini-2.5-flash"]

    def body(_: str) -> dict[str, Any]:
        return {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": temperature}}

    _, payload = _call(models, "generateContent", body, 45)
    try:
        parts = payload["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts).strip()
    except (KeyError, IndexError, TypeError) as e:
        raise AIUnavailable("empty answer") from e
    if not text:
        raise AIUnavailable("empty answer")
    return text


def pack(vec: array) -> bytes:
    return vec.tobytes()


def unpack(blob: Optional[bytes]) -> Optional[array]:
    if not blob:
        return None
    a = array("f")
    a.frombytes(blob)
    return a
