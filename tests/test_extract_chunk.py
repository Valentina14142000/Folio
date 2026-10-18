import io
import zipfile

import pytest

from app.services import chunker, extract

DOCX_XML = (
    '<?xml version="1.0"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
    "<w:p><w:r><w:t>Hello</w:t></w:r><w:r><w:t> world.</w:t></w:r></w:p><w:p><w:r><w:t>Second paragraph.</w:t></w:r></w:p>"
    "</w:body></w:document>"
)


def make_docx() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", DOCX_XML)
    return buf.getvalue()


def test_text_and_docx():
    assert extract.extract_pages("a.md", b"# Title\n\nSome text here.") == [(None, "# Title\n\nSome text here.")]
    assert extract.extract_pages("a.docx", make_docx()) == [(None, "Hello world.\n\nSecond paragraph.")]


@pytest.mark.parametrize(
    "name,data",
    [("a.exe", b"MZ"), ("a.pdf", b"nope"), ("a.txt", b""), ("a.docx", b"PK junk"), ("a.txt", b"bin\x00ary"), ("noext", b"x")],
)
def test_rejects_bad_files(name, data):
    with pytest.raises(extract.ExtractError):
        extract.extract_pages(name, data)


def test_clean_filename():
    assert extract.clean_filename("../../etc/passwd") == "passwd"
    assert extract.clean_filename("C:\\docs\\report.pdf") == "report.pdf"
    assert extract.clean_filename("") == "untitled"


def test_chunking_sizes_and_coverage():
    para = "Sentence number {} is here to fill space. "
    text = "\n\n".join("".join(para.format(i * 10 + j) for j in range(8)) for i in range(12))
    chunks = chunker.chunk_page(text)
    assert len(chunks) > 3
    assert all(len(c) <= chunker.TARGET + chunker.OVERLAP + 40 for c in chunks)
    joined = " ".join(chunks)
    assert "Sentence number 0 " in joined and "Sentence number 117 " in joined


def test_chunker_handles_one_giant_word_run():
    chunks = chunker.chunk_page("x" * 5000)
    assert chunks and all(len(c) <= chunker.TARGET + 5 for c in chunks)


def test_pdf_line_wraps_are_joined():
    wrapped = "Hotels are reimbursed up to 150 euros per night in major cities and\n110 euros elsewhere. Receipts are re-\nquired within 30 days.\n\nNew paragraph."
    out = extract._unwrap(wrapped)
    assert "cities and 110 euros" in out and "required within" in out and "\n\nNew paragraph." in out
