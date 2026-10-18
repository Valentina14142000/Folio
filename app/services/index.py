"""In-memory hybrid search: BM25 keyword ranking + cosine similarity on embeddings, fused with RRF."""
import math
import re
import threading
from array import array
from collections import Counter
from contextlib import closing
from dataclasses import dataclass, field
from typing import Optional

from app import db
from app.services import gemini

STOPWORDS = set(
    "a an and are as at be but by can do does for from had has have how i if in into is it its me my no not of on or our so such "
    "than that the their them then there these they this to was we were what when where which who why will with you your".split()
)
K1, B = 1.5, 0.75


def stem(w: str) -> str:
    for suf in ("ingly", "edly", "ing", "ed", "es", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def tokenize(text: str) -> list[str]:
    return [stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 1 and w not in STOPWORDS]


@dataclass
class Chunk:
    id: int
    doc_id: int
    doc_name: str
    page: Optional[int]
    text: str
    tf: Counter = field(default_factory=Counter)
    length: int = 0
    model: Optional[str] = None
    vec: Optional[array] = None


class Index:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.df: Counter = Counter()
        for c in chunks:
            self.df.update(c.tf.keys())
        self.avg_len = (sum(c.length for c in chunks) / len(chunks)) if chunks else 0.0

    def bm25(self, query: str, limit: int = 30) -> list[tuple[Chunk, float]]:
        terms = list(dict.fromkeys(tokenize(query)))
        if not terms or not self.chunks:
            return []
        n = len(self.chunks)
        scored = []
        for c in self.chunks:
            s = 0.0
            for t in terms:
                f = c.tf.get(t)
                if not f:
                    continue
                idf = math.log(1 + (n - self.df[t] + 0.5) / (self.df[t] + 0.5))
                s += idf * f * (K1 + 1) / (f + K1 * (1 - B + B * c.length / (self.avg_len or 1)))
            if s > 0:
                scored.append((c, s))
        scored.sort(key=lambda x: -x[1])
        return scored[:limit]

    def semantic(self, qvec: array, model: str, limit: int = 30) -> list[tuple[Chunk, float]]:
        scored = []
        for c in self.chunks:
            if c.vec is None or c.model != model or len(c.vec) != len(qvec):
                continue
            scored.append((c, sum(a * b for a, b in zip(qvec, c.vec))))
        scored.sort(key=lambda x: -x[1])
        return scored[:limit]


_lock = threading.Lock()
_index: Optional[Index] = None
_key: Optional[str] = None


def invalidate() -> None:
    global _index
    with _lock:
        _index = None


def get_index() -> Index:
    global _index, _key
    from app import config

    with _lock:
        path = config.database_path()
        if _index is None or _key != path:
            chunks: list[Chunk] = []
            with closing(db.connect()) as conn:
                rows = conn.execute(
                    "SELECT c.id, c.doc_id, d.name, c.page, c.text, c.embed_model, c.embedding "
                    "FROM chunks c JOIN documents d ON d.id = c.doc_id ORDER BY c.doc_id, c.ord"
                ).fetchall()
            for r in rows:
                toks = tokenize(r["text"])
                chunks.append(Chunk(
                    id=r["id"], doc_id=r["doc_id"], doc_name=r["name"], page=r["page"], text=r["text"],
                    tf=Counter(toks), length=len(toks), model=r["embed_model"], vec=gemini.unpack(r["embedding"]),
                ))
            _index, _key = Index(chunks), path
        return _index


def search(query: str, k: int = 5, semantic: bool = True) -> tuple[list[dict], str]:
    """Return (results, mode) where mode is 'hybrid' or 'keyword'."""
    idx = get_index()
    if not idx.chunks:
        return [], "keyword"
    ranked: dict[int, float] = {}
    lookup: dict[int, Chunk] = {}
    mode = "keyword"

    for rank, (c, _) in enumerate(idx.bm25(query), start=1):
        ranked[c.id] = ranked.get(c.id, 0.0) + 1.0 / (60 + rank)
        lookup[c.id] = c

    if semantic and any(c.vec is not None for c in idx.chunks):
        try:
            model, vecs = gemini.embed_texts([query], "RETRIEVAL_QUERY")
            for rank, (c, sim) in enumerate(idx.semantic(vecs[0], model), start=1):
                if sim < 0.3:
                    break
                ranked[c.id] = ranked.get(c.id, 0.0) + 1.0 / (60 + rank)
                lookup[c.id] = c
            mode = "hybrid"
        except gemini.AIUnavailable:
            pass

    top = sorted(ranked.items(), key=lambda x: -x[1])[:k]
    best = top[0][1] if top else 1.0
    return [{
        "chunk_id": cid,
        "doc_id": lookup[cid].doc_id,
        "doc_name": lookup[cid].doc_name,
        "page": lookup[cid].page,
        "text": lookup[cid].text,
        "score": round(score / best, 3),
    } for cid, score in top], mode
