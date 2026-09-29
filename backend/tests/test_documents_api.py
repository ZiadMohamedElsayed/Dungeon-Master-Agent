"""Documents API + misc endpoints, against an isolated temp Chroma."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_lore_upload_list_delete_roundtrip():
    body = b"Emberfall is a village beneath the Shadow Peaks. " * 20
    r = client.post("/documents/lore/upload", files={"file": ("t1.txt", body)})
    assert r.status_code == 200
    assert r.json()["chunks_added"] >= 1

    dup = client.post("/documents/lore/upload", files={"file": ("t1.txt", body)})
    assert dup.status_code == 409

    listing = client.get("/documents/lore/list").json()
    assert "t1.txt" in listing["documents"]
    row = next(d for d in listing["details"] if d["name"] == "t1.txt")
    assert row["chunks"] >= 1

    assert client.delete("/documents/lore/t1.txt").status_code == 200
    assert "t1.txt" not in client.get("/documents/lore/list").json()["documents"]
    assert client.delete("/documents/lore/t1.txt").status_code == 404


def test_upload_validation():
    bad_ext = client.post("/documents/lore/upload", files={"file": ("x.exe", b"zzz")})
    assert bad_ext.status_code == 400
    empty = client.post("/documents/lore/upload", files={"file": ("e.txt", b"")})
    assert empty.status_code == 400


def test_campaign_clear():
    body = b"Turn log filler. " * 20
    client.post("/documents/campaign/upload", files={"file": ("c1.txt", body)})
    assert client.delete("/documents/campaign/clear").status_code == 200
    assert client.get("/documents/campaign/list").json()["documents"] == []


def test_empty_query_rejected():
    assert client.post("/chat/", json={"query": ""}).status_code == 422


def test_cache_stats_shape():
    stats = client.get("/chat/cache/stats").json()
    assert set(stats) == {"embeddings", "retrieval_entries", "judge_entries", "epochs"}
    assert set(stats["epochs"]) == {"lore", "campaign"}
