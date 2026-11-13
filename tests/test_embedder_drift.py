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
    detector = DriftDetector(baseline_centroid=baseline, window_size=5, threshold=0.99, min_samples=1)

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
    detector = DriftDetector(baseline_centroid=baseline, window_size=5, threshold=0.99, min_samples=1)

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


# ---- warm-up, event cap, reset ------------------------------------------------

def _unit(dim=384, axis=0):
    v = np.zeros(dim, dtype=np.float32)
    v[axis] = 1.0
    return v


def test_no_drift_verdict_until_the_window_is_full():
    baseline = _unit(axis=0)
    detector = DriftDetector(baseline_centroid=baseline, window_size=4, threshold=0.9)
    off_topic = _unit(axis=1)  # orthogonal to the baseline: similarity 0

    for i in range(3):
        result = detector.add_embedding(off_topic)
        assert result["warming_up"] is True
        assert result["drifted"] is False
    assert detector.drift_events == [] and detector.total_events == 0

    result = detector.add_embedding(off_topic)  # fourth embedding fills the window
    assert result["warming_up"] is False
    assert result["drifted"] is True
    assert detector.total_events == 1


def test_similarity_is_still_reported_while_warming_up():
    detector = DriftDetector(baseline_centroid=_unit(axis=0), window_size=10, threshold=0.9)
    result = detector.add_embedding(_unit(axis=0))
    assert result["warming_up"] is True
    assert abs(result["similarity"] - 1.0) < 1e-6


def test_min_samples_lets_you_judge_earlier_and_is_clamped_to_the_window():
    off = _unit(axis=1)
    early = DriftDetector(_unit(axis=0), window_size=10, threshold=0.9, min_samples=2)
    assert early.add_embedding(off)["drifted"] is False
    assert early.add_embedding(off)["drifted"] is True

    too_big = DriftDetector(_unit(axis=0), window_size=3, threshold=0.9, min_samples=99)
    assert too_big.min_samples == 3
    too_small = DriftDetector(_unit(axis=0), window_size=3, threshold=0.9, min_samples=0)
    assert too_small.min_samples == 1


def test_event_history_is_capped_but_the_total_keeps_counting():
    detector = DriftDetector(_unit(axis=0), window_size=1, threshold=0.9, max_events=5)
    off = _unit(axis=1)
    for _ in range(12):
        detector.add_embedding(off)
    assert len(detector.drift_events) == 5
    assert detector.total_events == 12


def test_clear_events_resets_the_list_and_the_total():
    detector = DriftDetector(_unit(axis=0), window_size=1, threshold=0.9)
    detector.add_embedding(_unit(axis=1))
    assert detector.total_events == 1
    detector.clear_events()
    assert detector.drift_events == [] and detector.total_events == 0
