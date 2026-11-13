"""
Embedding drift detection.

Tracks a rolling window of recent query/document embeddings and compares their
centroid against a baseline centroid (computed at index time) using cosine
similarity. When similarity falls below a threshold, a drift event fires and
callers can trigger automated re-indexing.
"""
from collections import deque
import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class DriftDetector:
    def __init__(self, baseline_centroid: np.ndarray, window_size: int = 50,
                 threshold: float = 0.85, min_samples: int | None = None,
                 max_events: int = 1000):
        self.baseline_centroid = baseline_centroid
        self.window: deque[np.ndarray] = deque(maxlen=window_size)
        self.threshold = threshold
        # No drift verdict until this many embeddings are in the window. With the
        # default (a full window) the centroid is a real sample of recent traffic; a
        # window of one or two queries is never close to the baseline centroid, so
        # judging it would raise a false alarm on the first requests after every start
        # or reset. Clamped to 1..window_size.
        wanted = window_size if min_samples is None else min_samples
        self.min_samples = max(1, min(wanted, window_size))
        # Only the most recent max_events events are kept (a long drift would otherwise
        # add one per query forever); total_events keeps counting past the cap.
        self.max_events = max_events
        self.drift_events: list[dict] = []
        self.total_events = 0

    def add_embedding(self, vector: np.ndarray) -> dict:
        self.window.append(vector)
        current_centroid = np.mean(np.stack(self.window), axis=0)
        similarity = cosine_similarity(self.baseline_centroid, current_centroid)
        warming_up = len(self.window) < self.min_samples
        drifted = (not warming_up) and similarity < self.threshold

        result = {
            "similarity": similarity,
            "threshold": self.threshold,
            "drifted": drifted,
            "warming_up": warming_up,
            "window_size": len(self.window),
        }

        if drifted:
            self.total_events += 1
            self.drift_events.append(result)
            if len(self.drift_events) > self.max_events:
                del self.drift_events[: len(self.drift_events) - self.max_events]

        return result

    def clear_events(self):
        """Forget recorded drift events and the running total (used by the demo reset)."""
        self.drift_events.clear()
        self.total_events = 0

    def reindex_baseline(self):
        """Call after re-indexing to reset baseline to the current window centroid."""
        if self.window:
            self.baseline_centroid = np.mean(np.stack(self.window), axis=0)
