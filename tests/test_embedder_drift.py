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


def test_reindex_baseline_updates_to_window_centroid():
    baseline = np.ones(384, dtype=np.float32) / np.sqrt(384)
    detector = DriftDetector(baseline_centroid=baseline, window_size=5, threshold=0.99)

    new_vec = -baseline
    detector.add_embedding(new_vec)
    detector.reindex_baseline()

    assert np.allclose(detector.baseline_centroid, new_vec)


def test_reindex_baseline_noop_when_window_empty():
    baseline = np.ones(384, dtype=np.float32) / np.sqrt(384)
    detector = DriftDetector(baseline_centroid=baseline, window_size=5, threshold=0.99)

    detector.reindex_baseline()

    assert np.allclose(detector.baseline_centroid, baseline)


def test_drift_events_accumulate_only_on_drift():
    baseline = np.ones(384, dtype=np.float32) / np.sqrt(384)
    detector = DriftDetector(baseline_centroid=baseline, window_size=5, threshold=0.99)

    detector.add_embedding(baseline)
    assert len(detector.drift_events) == 0

    detector.add_embedding(-baseline)
    assert len(detector.drift_events) == 1


def test_embed_batch_returns_correct_shape():
    embedder = NgramEmbedder(dim=384)
    result = embedder.embed_batch(["hello", "world", "test"])
    assert result.shape == (3, 384)


def test_embedder_handles_empty_string():
    embedder = NgramEmbedder(dim=384)
    v = embedder.embed("")
    assert v.shape == (384,)
    assert np.linalg.norm(v) == 0.0


def test_embedder_handles_text_shorter_than_ngram_size():
    embedder = NgramEmbedder(dim=384, ngram_size=3)
    v = embedder.embed("ab")
    assert v.shape == (384,)
    assert np.linalg.norm(v) > 0.0


def test_embedder_different_texts_produce_different_vectors():
    embedder = NgramEmbedder(dim=384)
    v1 = embedder.embed("hello world")
    v2 = embedder.embed("completely different text")
    assert not np.allclose(v1, v2)


def test_cosine_similarity_zero_vector_returns_zero():
    a = np.zeros(3)
    b = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(a, b) == 0.0
