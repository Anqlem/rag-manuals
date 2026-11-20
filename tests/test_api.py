from fastapi.testclient import TestClient

from rag.api import app

client = TestClient(app)


def test_ask_and_search(ingested):
    assert client.get("/health").json()["documents"] >= 1
    r = client.post("/ask", json={"question": "Which oil goes in the gearbox?"})
    assert r.status_code == 200 and r.json()["sources"]
    r = client.post("/search", json={"question": "fan part number", "strategy": "fixed"})
    assert r.status_code == 200 and r.json()["hits"]


def test_rejects_bad_mode():
    r = client.post("/ask", json={"question": "anything", "mode": "nope"})
    assert r.status_code == 422


def test_ui_served():
    assert "Manuals Q&amp;A" in client.get("/").text
