def test_health_and_ui(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/api/health").json()["ai_configured"] is False
    r = client.get("/")
    assert r.status_code == 200 and "Folio" in r.text


def test_upload_search_chat_delete(client):
    body = b"Our refund policy: customers can return items within 30 days for a full refund. Shipping costs are not refunded."
    r = client.post("/api/documents?filename=policy.txt", content=body, headers={"content-type": "application/octet-stream"})
    assert r.status_code == 201
    doc = r.json()
    assert doc["name"] == "policy.txt" and doc["chunk_count"] >= 1

    assert [d["id"] for d in client.get("/api/documents").json()] == [doc["id"]]
    preview = client.get(f"/api/documents/{doc['id']}").json()["preview"]
    assert "refund" in preview[0]["text"]

    hits = client.post("/api/search", json={"query": "refund within 30 days"}).json()
    assert hits["mode"] == "keyword" and hits["results"][0]["doc_name"] == "policy.txt"

    chat = client.post("/api/chat", json={"question": "How long do I have to return items?"}).json()
    assert chat["mode"] == "extractive" and chat["sources"] and "30 days" in chat["answer"]

    assert client.delete(f"/api/documents/{doc['id']}").status_code == 204
    assert client.get(f"/api/documents/{doc['id']}").status_code == 404
    assert client.post("/api/chat", json={"question": "refund?"}).json()["mode"] == "none"


def test_upload_errors(client):
    h = {"content-type": "application/octet-stream"}
    assert client.post("/api/documents?filename=a.exe", content=b"MZ", headers=h).status_code == 400
    assert client.post("/api/documents?filename=a.pdf", content=b"not pdf", headers=h).status_code == 400
    assert client.post("/api/documents?filename=a.txt", content=b"", headers=h).status_code == 400
    assert client.post("/api/documents", content=b"x", headers=h).status_code == 422  # filename required


def test_oversize_upload_rejected(client, monkeypatch):
    from app import config

    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 100)
    r = client.post("/api/documents?filename=big.txt", content=b"a" * 500, headers={"content-type": "application/octet-stream"})
    assert r.status_code == 413


def test_text_and_samples(client):
    assert client.post("/api/documents/text", json={"title": "Notes", "text": "short"}).status_code == 400
    ok = client.post("/api/documents/text", json={"title": "Notes", "text": "Meeting notes: launch is planned for March."})
    assert ok.status_code == 201
    assert client.post("/api/documents/samples").json() == {"created": 2}
    assert client.get("/api/stats").json()["documents"] == 3


def test_validation(client):
    assert client.post("/api/chat", json={"question": ""}).status_code == 422
    assert client.post("/api/chat", json={"question": "ok?", "history": [{"role": "system", "content": "x"}]}).status_code == 422
    assert client.post("/api/search", json={"query": "a"}).status_code == 422


def test_followup_uses_previous_question(client):
    client.post("/api/documents/samples")
    first = {"role": "user", "content": "How many sick days do I get?"}
    r = client.post("/api/chat", json={"question": "and parental?", "history": [first, {"role": "assistant", "content": "10 days"}]})
    assert r.status_code == 200 and r.json()["sources"]


def test_rate_limit(client):
    codes = [client.post("/api/search", json={"query": "anything"}).status_code for _ in range(45)]
    assert codes[0] == 200 and 429 in codes
