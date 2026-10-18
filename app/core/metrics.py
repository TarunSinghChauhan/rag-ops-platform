"""In-memory metrics for the ops dashboard."""
from collections import deque
import numpy as np


class LatencyTracker:
    def __init__(self, maxlen: int = 1000):
        self.samples = deque(maxlen=maxlen)

    def record(self, ms: float):
        self.samples.append(ms)

    def summary(self) -> dict:
        if not self.samples:
            return {"count": 0, "p50": None, "p95": None, "p99": None, "recent": []}
        arr = np.array(self.samples, dtype=float)
        return {
            "count": len(arr),
            "p50": round(float(np.percentile(arr, 50)), 2),
            "p95": round(float(np.percentile(arr, 95)), 2),
            "p99": round(float(np.percentile(arr, 99)), 2),
            "recent": [round(float(x), 2) for x in list(self.samples)[-60:]],
        }


latency = LatencyTracker()
drift_history: deque = deque(maxlen=100)
