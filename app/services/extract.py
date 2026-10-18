"""Turn uploaded files into plain text pages. Nothing is written to disk."""
import io
import re
import zipfile
from xml.etree import ElementTree

TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".csv", ".log", ".json", ".rst"}
ALLOWED = {".pdf", ".docx"} | TEXT_EXTENSIONS


class ExtractError(ValueError):
    """The file cannot be read; the message is safe to show to the user."""


def extension(filename: str) -> str:
    m = re.search(r"(\.[A-Za-z0-9]+)$", filename or "")
    return m.group(1).lower() if m else ""


def clean_filename(filename: str) -> str:
    name = (filename or "").replace("\\", "/").split("/")[-1]
    name = re.sub(r"[\x00-\x1f]", "", name).strip()
    return name[:150] or "untitled"


def _unwrap(text: str) -> str:
    """PDF text comes with a hard line break at the end of every visual line; join them back into paragraphs."""
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)          # re-join words hyphenated across lines
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text.strip())      # single newline -> space (blank lines stay)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def _pdf(data: bytes) -> list[tuple[int | None, str]]:
    if not data.startswith(b"%PDF"):
        raise ExtractError("This file is not a valid PDF.")
    try:
        from pypdf import PdfReader
    except ImportError as e:  # pragma: no cover
        raise ExtractError("PDF support is not installed on the server.") from e
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                raise ExtractError("This PDF is password protected.")
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            text = _unwrap(page.extract_text() or "")
            if text:
                pages.append((i, text))
    except ExtractError:
        raise
    except Exception as e:
        raise ExtractError("Could not read this PDF. It may be damaged.") from e
    if not pages:
        raise ExtractError("No text found. Scanned PDFs (images) are not supported yet.")
    return pages


def _docx(data: bytes) -> list[tuple[int | None, str]]:
    if not data.startswith(b"PK"):
        raise ExtractError("This file is not a valid .docx document.")
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            if sum(i.file_size for i in z.infolist()) > 80 * 1024 * 1024:
                raise ExtractError("Document is too large when unpacked.")
            xml = z.read("word/document.xml")
    except ExtractError:
        raise
    except Exception as e:
        raise ExtractError("Could not read this .docx file.") from e
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    root = ElementTree.fromstring(xml)
    paragraphs = []
    for p in root.iter(ns + "p"):
        text = "".join(t.text or "" for t in p.iter(ns + "t")).strip()
        if text:
            paragraphs.append(text)
    if not paragraphs:
        raise ExtractError("No text found in this document.")
    return [(None, "\n\n".join(paragraphs))]


def _text(data: bytes) -> list[tuple[int | None, str]]:
    for enc in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            text = data.decode(enc)
            break
        except UnicodeError:
            continue
    else:  # pragma: no cover
        raise ExtractError("Could not decode this text file.")
    if "\x00" in text:
        raise ExtractError("This looks like a binary file, not text.")
    text = text.strip()
    if not text:
        raise ExtractError("The file is empty.")
    return [(None, text)]


def extract_pages(filename: str, data: bytes) -> list[tuple[int | None, str]]:
    ext = extension(filename)
    if ext not in ALLOWED:
        raise ExtractError("Unsupported file type. Use PDF, DOCX, TXT, MD, CSV, JSON or LOG.")
    if not data:
        raise ExtractError("The file is empty.")
    if ext == ".pdf":
        return _pdf(data)
    if ext == ".docx":
        return _docx(data)
    return _text(data)
