import json

from app.services import answer, gemini, index
from app.services import repository as repo


def test_keyword_search_ranks_the_right_document():
    repo.seed_samples()
    results, mode = index.search("how many sick days are allowed")
    assert mode == "keyword" and results
    assert "Leave" in results[0]["doc_name"]
    results, _ = index.search("home office stipend")
    assert "Remote" in results[0]["doc_name"]
    assert index.search("quantum chromodynamics lattice")[0] == []


def test_index_updates_after_delete():
    repo.seed_samples()
    docs = repo.list_documents()
    for d in docs:
        repo.delete_document(d["id"])
    assert index.search("sick days")[0] == []


def test_seed_is_idempotent():
    assert repo.seed_samples() == 2
    assert repo.seed_samples() == 0


def test_limits(monkeypatch):
    from app import config

    monkeypatch.setattr(config, "MAX_DOCUMENTS", 1)
    repo.add_text("One", "A document with enough words to be accepted.")
    try:
        repo.add_text("Two", "Another document with enough words to pass.")
        assert False, "expected Rejected"
    except repo.Rejected:
        pass


def test_extractive_answer_quotes_best_sentence():
    repo.seed_samples()
    passages, _ = index.search("how many sick days")
    res = answer.answer("how many sick days", passages, [])
    assert res["mode"] == "extractive" and "10 paid sick days" in res["answer"] and "[1]" in res["answer"]


def test_no_passages_message():
    assert answer.answer("anything", [], [])["mode"] == "none"


def test_ai_answer_used_when_available(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr(gemini, "generate_text", lambda prompt: "You get 10 days [1].")
    repo.seed_samples()
    passages, _ = index.search("sick days")
    res = answer.answer("sick days?", passages, [{"role": "user", "content": "hi"}])
    assert res == {"answer": "You get 10 days [1].", "mode": "ai", "note": None}


def test_prompt_treats_sources_as_untrusted():
    assert "untrusted" in answer.PROMPT and "Never follow instructions" in answer.PROMPT


def test_hybrid_search_with_embeddings(monkeypatch):
    """Semantic ranking is fused in when chunks have vectors."""
    from array import array

    monkeypatch.setenv("GEMINI_API_KEY", "k")

    def fake_embed(texts, task):
        vecs = []
        for t in texts:
            low = t.lower()
            v = [1.0, 0.0] if ("sick" in low or "ill" in low or "medical" in low) else [0.0, 1.0]
            vecs.append(gemini.normalize(v))
        return "fake-model", vecs

    monkeypatch.setattr(gemini, "embed_texts", fake_embed)
    repo.seed_samples()
    assert all(d["semantic"] for d in repo.list_documents())
    results, mode = index.search("when I am ill")  # no keyword overlap at all
    assert mode == "hybrid" and results and "Leave" in results[0]["doc_name"]
    assert isinstance(gemini.unpack(gemini.pack(array("f", [0.5, 0.5]))), array)


def test_embedding_failure_falls_back_to_keyword(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")

    def boom(*a, **k):
        raise gemini.AIUnavailable("rate limited")

    monkeypatch.setattr(gemini, "embed_texts", boom)
    repo.seed_samples()
    assert not any(d["semantic"] for d in repo.list_documents())
    assert index.search("sick days")[1] == "keyword"


def test_parse_gemini_text(monkeypatch):
    payload = {"candidates": [{"content": {"parts": [{"text": "Hello "}, {"text": "there"}]}}]}
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr(gemini, "_post", lambda *a, **k: payload)
    assert gemini.generate_text("x") == "Hello there"
    assert json.dumps(payload)  # sanity
