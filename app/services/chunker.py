"""Split text into overlapping, paragraph-aware chunks."""
import re

TARGET = 800
OVERLAP = 150


def _split_long(paragraph: str) -> list[str]:
    """Break an oversized paragraph on sentence boundaries, then on whitespace."""
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    out, cur = [], ""
    for s in sentences:
        while len(s) > TARGET:  # a single huge "sentence": hard split at whitespace
            cut = s.rfind(" ", 0, TARGET)
            cut = cut if cut > TARGET // 2 else TARGET
            if cur:
                out.append(cur)
                cur = ""
            out.append(s[:cut].strip())
            s = s[cut:].strip()
        if cur and len(cur) + 1 + len(s) > TARGET:
            out.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        out.append(cur)
    return out


def chunk_page(text: str) -> list[str]:
    paragraphs = [re.sub(r"[ \t]+", " ", p).strip() for p in re.split(r"\n\s*\n|\r\n\s*\r\n", text)]
    pieces: list[str] = []
    for p in paragraphs:
        if not p:
            continue
        pieces.extend(_split_long(p) if len(p) > TARGET else [p])

    chunks: list[str] = []
    cur = ""
    for piece in pieces:
        if cur and len(cur) + 2 + len(piece) > TARGET:
            chunks.append(cur)
            tail = cur[-OVERLAP:]
            tail = tail[tail.find(" ") + 1:] if " " in tail else tail
            cur = f"{tail}\n\n{piece}" if len(tail) + len(piece) < TARGET else piece
        else:
            cur = f"{cur}\n\n{piece}" if cur else piece
    if cur.strip():
        chunks.append(cur)
    return [c.strip() for c in chunks if len(c.strip()) >= 20 or len(chunks) == 1]


def chunk_pages(pages: list[tuple[int | None, str]]) -> list[tuple[int | None, str]]:
    out: list[tuple[int | None, str]] = []
    for page, text in pages:
        for c in chunk_page(text):
            out.append((page, c))
    return out
