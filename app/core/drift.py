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
                 threshold: float = 0.85):
        self.baseline_centroid = baseline_centroid
        self.window: deque[np.ndarray] = deque(maxlen=window_size)
        self.threshold = threshold
        self.drift_events: list[dict] = []

    def add_embedding(self, vector: np.ndarray) -> dict:
        self.window.append(vector)
        current_centroid = np.mean(np.stack(self.window), axis=0)
        similarity = cosine_similarity(self.baseline_centroid, current_centroid)
        drifted = similarity < self.threshold

        result = {
            "similarity": similarity,
            "threshold": self.threshold,
            "drifted": drifted,
            "window_size": len(self.window),
        }

        if drifted:
            self.drift_events.append(result)

        return result

    def reindex_baseline(self):
        """Call after re-indexing to reset baseline to the current window centroid."""
        if self.window:
            self.baseline_centroid = np.mean(np.stack(self.window), axis=0)
