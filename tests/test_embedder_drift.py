import numpy as np
from app.core.embedder import NgramEmbedder
from app.core.drift import DriftDetector, cosine_similarity


def test_embedder_is_deterministic():
    embedder = NgramEmbedder(dim=384)
    v1 = embedder.embed("hello world")
    v2 = embedder.embed("hello world")
    assert np.allclose(v1, v2)


def test_embedder_output_is_normalized():
    embedder = NgramEmbedder(dim=384)
    v = embedder.embed("some sample text for embedding")
    norm = np.linalg.norm(v)
    assert abs(norm - 1.0) < 1e-5 or norm == 0.0


def test_drift_detector_flags_low_similarity():
    baseline = np.ones(384, dtype=np.float32) / np.sqrt(384)
    detector = DriftDetector(baseline_centroid=baseline, window_size=5, threshold=0.99)

    opposite = -baseline
    result = detector.add_embedding(opposite)

    assert result["drifted"] is True
    assert result["similarity"] < 0.99


def test_cosine_similarity_identical_vectors_is_one():
    v = np.array([1.0, 2.0, 3.0])
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-6
