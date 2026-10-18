from contextlib import closing
from datetime import datetime, timezone
from typing import Any, Optional

from app import config, db
from app.services import chunker, extract, gemini, index


class NotFound(Exception):
    pass


class Rejected(Exception):
    """The request is valid but cannot be accepted (limits, empty content)."""


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def list_documents() -> list[dict[str, Any]]:
    with closing(db.connect()) as conn:
        rows = conn.execute(
            "SELECT d.*, (SELECT COUNT(*) FROM chunks c WHERE c.doc_id = d.id AND c.embedding IS NOT NULL) AS embedded_chunks "
            "FROM documents d ORDER BY d.created_at DESC, d.id DESC"
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["semantic"] = d["embedded_chunks"] == d["chunk_count"] and d["chunk_count"] > 0
        out.append(d)
    return out


def get_document(doc_id: int, preview: int = 6) -> dict[str, Any]:
    with closing(db.connect()) as conn:
        row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        if row is None:
            raise NotFound("Document not found.")
        chunks = conn.execute(
            "SELECT id, ord, page, text FROM chunks WHERE doc_id = ? ORDER BY ord LIMIT ?", (doc_id, preview)
        ).fetchall()
    return {**dict(row), "preview": [dict(c) for c in chunks]}


def delete_document(doc_id: int) -> None:
    with closing(db.connect()) as conn:
        cur = conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.commit()
        if cur.rowcount == 0:
            raise NotFound("Document not found.")
    index.invalidate()


def add_document(name: str, kind: str, size_bytes: int, pages: list[tuple[Optional[int], str]], embed: bool = True) -> dict[str, Any]:
    total_chars = sum(len(t) for _, t in pages)
    if total_chars > config.MAX_CHARS_PER_DOC:
        raise Rejected(f"Document is too long ({total_chars:,} characters; the limit is {config.MAX_CHARS_PER_DOC:,}).")
    chunks = chunker.chunk_pages(pages)
    if not chunks:
        raise Rejected("No readable text found in this document.")

    with closing(db.connect()) as conn:
        n_docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        n_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    if n_docs >= config.MAX_DOCUMENTS:
        raise Rejected(f"Document limit reached ({config.MAX_DOCUMENTS}). Delete one first.")
    if n_chunks + len(chunks) > config.MAX_TOTAL_CHUNKS:
        raise Rejected("The knowledge base is full. Delete a document first.")

    model: Optional[str] = None
    vectors: list = []
    if embed and config.gemini_key():
        try:
            model, vectors = gemini.embed_texts([c for _, c in chunks], "RETRIEVAL_DOCUMENT")
        except gemini.AIUnavailable:
            model, vectors = None, []  # keyword search still works

    with closing(db.connect()) as conn:
        cur = conn.execute(
            "INSERT INTO documents (name, kind, size_bytes, chunk_count, char_count, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (name, kind, size_bytes, len(chunks), total_chars, now_iso()),
        )
        doc_id = cur.lastrowid
        conn.executemany(
            "INSERT INTO chunks (doc_id, ord, page, text, embed_model, embedding) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (doc_id, i, page, text, model if vectors else None, gemini.pack(vectors[i]) if vectors else None)
                for i, (page, text) in enumerate(chunks)
            ],
        )
        conn.commit()
    index.invalidate()
    return get_document(doc_id, preview=0)


def add_file(filename: str, data: bytes) -> dict[str, Any]:
    name = extract.clean_filename(filename)
    pages = extract.extract_pages(name, data)
    return add_document(name, extract.extension(name).lstrip("."), len(data), pages)


def add_text(title: str, text: str) -> dict[str, Any]:
    text = text.strip()
    if len(text) < 20:
        raise Rejected("Add at least a sentence or two of text.")
    return add_document(extract.clean_filename(title) or "Pasted text", "text", len(text.encode("utf-8")), [(None, text)])


def stats() -> dict[str, Any]:
    with closing(db.connect()) as conn:
        d = conn.execute("SELECT COUNT(*), COALESCE(SUM(char_count), 0) FROM documents").fetchone()
        c = conn.execute("SELECT COUNT(*), SUM(embedding IS NOT NULL) FROM chunks").fetchone()
    return {"documents": d[0], "characters": d[1], "chunks": c[0], "embedded_chunks": c[1] or 0}


SAMPLES = [
    ("Remote Work Policy (sample)", """Remote Work Policy

Eligibility. Full-time employees who have completed their 90-day probation may work remotely up to three days per week. Roles that require on-site equipment, such as lab technicians and facilities staff, are not eligible.

Core hours. Everyone is expected to be reachable between 10:00 and 15:00 in their local time zone. Meetings should be scheduled inside these hours whenever possible.

Equipment and expenses. The company provides a laptop, a monitor and a headset. Employees receive a one-time home office stipend of 400 euros and a monthly internet allowance of 30 euros. Receipts must be submitted within 60 days.

Security. Work must be done on company-managed devices over the VPN. Public Wi-Fi may only be used with the VPN enabled. Lost or stolen devices must be reported to IT within 24 hours.

Approval. Request remote days in the HR portal at least one week ahead. Managers respond within two working days. Fully remote arrangements need approval from a department head and are reviewed every six months."""),
    ("Leave and Time Off (sample)", """Leave and Time Off

Annual leave. Employees receive 25 days of paid annual leave per calendar year, plus public holidays. Up to 5 unused days can be carried over to the next year and must be used by 31 March.

Sick leave. Employees may take up to 10 paid sick days per year without a doctor's note. A medical certificate is required after three consecutive sick days.

Parental leave. Birth parents receive 20 weeks of fully paid leave. Non-birth parents receive 8 weeks of fully paid leave. Leave can be taken flexibly within the first 12 months.

Requesting leave. Submit requests in the HR portal. Leave of more than 10 consecutive days must be requested at least one month in advance.

Public speaking and volunteering. Every employee has two paid volunteering days per year, which do not count toward annual leave."""),
]


def seed_samples() -> int:
    n = 0
    existing = {d["name"] for d in list_documents()}
    for title, text in SAMPLES:
        if title in existing:
            continue
        add_document(title, "text", len(text.encode("utf-8")), [(None, text)])
        n += 1
    return n
