"""An unseeded (zero) drift baseline means "no verdict", not "similarity 0 = drifted"."""
import numpy as np
import pytest

from app.api import dashboard
from app.api import routes as routes_module
from app.core.drift import DriftDetector


def unit(dim=8, axis=0):
    v = np.zeros(dim, dtype=np.float32)
    v[axis] = 1.0
    return v


# ── DriftDetector ─────────────────────────────────────────────────────────────────

def test_zero_baseline_is_not_ready():
    assert DriftDetector(np.zeros(8, dtype=np.float32)).baseline_ready is False
    assert DriftDetector(unit()).baseline_ready is True


def test_zero_baseline_never_reports_drift_even_with_a_full_window():
    d = DriftDetector(np.zeros(8, dtype=np.float32), window_size=3, threshold=0.85)
    for _ in range(5):
        r = d.add_embedding(unit())
    assert r["drifted"] is False
    assert r["similarity"] is None
    assert r["baseline_unset"] is True
    assert d.total_events == 0 and d.drift_events == []


def test_zero_baseline_still_tracks_the_window():
    d = DriftDetector(np.zeros(8, dtype=np.float32), window_size=3)
    d.add_embedding(unit())
    r = d.add_embedding(unit())
    assert r["window_size"] == 2


def test_a_seeded_baseline_still_detects_drift():
    d = DriftDetector(unit(axis=0), window_size=3, threshold=0.85)
    for _ in range(3):
        r = d.add_embedding(unit(axis=1))
    assert r["drifted"] is True
    assert r["similarity"] == pytest.approx(0.0)
    assert "baseline_unset" not in r


def test_a_seeded_baseline_with_matching_queries_is_not_drift():
    d = DriftDetector(unit(axis=0), window_size=3, threshold=0.85)
    for _ in range(3):
        r = d.add_embedding(unit(axis=0))
    assert r["drifted"] is False
    assert r["similarity"] == pytest.approx(1.0)


def test_reindexing_from_a_window_makes_the_baseline_ready():
    d = DriftDetector(np.zeros(8, dtype=np.float32), window_size=3)
    d.add_embedding(unit(axis=2))
    assert d.reindex_baseline() is True
    assert d.baseline_ready is True


def test_reindex_baseline_reports_a_noop_on_an_empty_window():
    base = unit(axis=0)
    d = DriftDetector(base, window_size=3)
    assert d.reindex_baseline() is False
    assert np.allclose(d.baseline_centroid, base)


# ── startup seeding ───────────────────────────────────────────────────────────────

@pytest.fixture
def zeroed_detector():
    det = dashboard._drift_detector
    saved = det.baseline_centroid
    det.baseline_centroid = np.zeros(dashboard.embedder.dim, dtype=np.float32)
    yield det
    det.baseline_centroid = saved


def test_baseline_is_set_even_when_the_vector_store_is_down(monkeypatch, zeroed_detector):
    def down():
        raise RuntimeError("qdrant unreachable")

    monkeypatch.setattr(dashboard.vector_store, "count", down)
    dashboard.seed_if_empty()
    assert zeroed_detector.baseline_ready is True


def test_baseline_is_set_on_a_normal_startup(monkeypatch, zeroed_detector):
    monkeypatch.setattr(dashboard.vector_store, "count", lambda: 20)
    monkeypatch.setattr(dashboard, "run_eval", lambda: None)
    dashboard.seed_if_empty()
    assert zeroed_detector.baseline_ready is True


# ── routes ────────────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


def test_reindex_baseline_with_an_empty_window_is_a_409_not_a_fake_success(client, monkeypatch):
    monkeypatch.setattr(routes_module._drift_detector, "reindex_baseline", lambda: False)
    resp = client.post("/reindex-baseline")
    assert resp.status_code == 409
    assert "unchanged" in resp.json()["detail"]


def test_reindex_baseline_with_queries_in_the_window_succeeds(client, monkeypatch):
    monkeypatch.setattr(routes_module._drift_detector, "reindex_baseline", lambda: True)
    resp = client.post("/reindex-baseline")
    assert resp.status_code == 200
    assert resp.json() == {"status": "baseline reset"}


def test_query_with_an_unset_baseline_adds_no_fake_zero_to_the_history(client, monkeypatch):
    from app.core.metrics import drift_history

    det = routes_module._drift_detector
    saved = det.baseline_centroid
    det.baseline_centroid = np.zeros(routes_module.embedder.dim, dtype=np.float32)
    drift_history.clear()
    monkeypatch.setattr(routes_module.cache, "get", lambda q, k=5: {"results": []})
    try:
        resp = client.post("/query", json={"query": "hello world", "top_k": 3})
    finally:
        det.baseline_centroid = saved
        det.window.clear()
    assert resp.status_code == 200
    body = resp.json()
    assert body["drift"]["similarity"] is None and body["drift"]["drifted"] is False
    assert list(drift_history) == []


def test_metrics_reports_unknown_similarity_when_the_baseline_is_unset(client):
    det = dashboard._drift_detector
    saved = det.baseline_centroid
    det.baseline_centroid = np.zeros(dashboard.embedder.dim, dtype=np.float32)
    det.window.append(unit(dim=dashboard.embedder.dim))
    try:
        body = client.get("/metrics").json()
    finally:
        det.baseline_centroid = saved
        det.window.clear()
    assert body["drift"]["current_similarity"] is None
