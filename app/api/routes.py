from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from app import config
from app.services import answer as answer_service
from app.services import extract, index
from app.services import repository as repo
from app.services.ratelimit import limited

router = APIRouter(prefix="/api")


def _guard(request: Request, bucket: str, max_calls: int) -> None:
    ip = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip() or (request.client.host if request.client else "local")
    if limited(f"{bucket}:{ip}", max_calls=max_calls):
        raise HTTPException(status_code=429, detail="Too many requests. Please wait a minute.")


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=4000)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=1000)
    history: list[Message] = Field(default_factory=list, max_length=20)

    @field_validator("question")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 2:
            raise ValueError("Ask a question.")
        return v


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=500)
    k: int = Field(8, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 2:
            raise ValueError("Enter a search term.")
        return v


class TextDocument(BaseModel):
    title: str = Field("Pasted text", max_length=150)
    text: str = Field(..., min_length=1, max_length=config.MAX_CHARS_PER_DOC)


def _wrap(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except repo.NotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except (repo.Rejected, extract.ExtractError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/health")
def health():
    return {"status": "ok", "ai_configured": bool(config.gemini_key())}


@router.get("/documents")
def list_documents():
    return repo.list_documents()


@router.post("/documents", status_code=201)
async def upload_document(request: Request, filename: str = Query(..., min_length=1, max_length=200)):
    """Upload a file as the raw request body; the file name goes in the query string."""
    _guard(request, "upload", 12)
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File is too large (limit {config.MAX_UPLOAD_BYTES // 1024 // 1024} MB).")
    data = await request.body()
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File is too large (limit {config.MAX_UPLOAD_BYTES // 1024 // 1024} MB).")
    return await run_in_threadpool(_wrap, repo.add_file, filename, data)


@router.post("/documents/text", status_code=201)
async def add_text(payload: TextDocument, request: Request):
    _guard(request, "upload", 12)
    return await run_in_threadpool(_wrap, repo.add_text, payload.title, payload.text)


@router.post("/documents/samples", status_code=201)
async def add_samples(request: Request):
    _guard(request, "upload", 6)
    return {"created": await run_in_threadpool(_wrap, repo.seed_samples)}


@router.get("/documents/{doc_id}")
def get_document(doc_id: int):
    return _wrap(repo.get_document, doc_id)


@router.delete("/documents/{doc_id}", status_code=204)
def delete_document(doc_id: int):
    _wrap(repo.delete_document, doc_id)
    return Response(status_code=204)


@router.post("/search")
def search(payload: SearchRequest, request: Request):
    _guard(request, "ai", 40)
    results, mode = index.search(payload.query, k=payload.k)
    return {"mode": mode, "results": results}


@router.post("/chat")
def chat(payload: ChatRequest, request: Request):
    _guard(request, "ai", 20)
    history = [m.model_dump() for m in payload.history]
    query = payload.question
    prev_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
    if prev_user and len(index.tokenize(query)) < 4:
        query = f"{prev_user} {query}"  # short follow-up: keep the topic of the previous question
    passages, mode = index.search(query, k=6)
    result = answer_service.answer(payload.question, passages, history)
    for p in passages:
        p["text"] = p["text"][:900]
    return {**result, "retrieval": mode, "sources": passages}


@router.get("/stats")
def stats():
    return repo.stats()
