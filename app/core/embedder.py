"""
Deterministic n-gram embedding.

No external API calls, no API keys, no network dependency. Text is hashed into
a fixed-dimension vector using character n-grams + a stable hash function, then
L2-normalized. This is intentionally simple (not semantically rich like a real
transformer embedding) but is fully reproducible, free, and good enough to
demonstrate RAG infra (indexing, retrieval, drift detection) without any billing
surface.
"""
import hashlib
import numpy as np


class NgramEmbedder:
    def __init__(self, dim: int = 384, ngram_size: int = 3):
        self.dim = dim
        self.ngram_size = ngram_size

    def _ngrams(self, text: str) -> list[str]:
        text = text.lower().strip()
        if len(text) < self.ngram_size:
            return [text] if text else []
        return [text[i:i + self.ngram_size] for i in range(len(text) - self.ngram_size + 1)]

    def _hash_to_index(self, ngram: str) -> int:
        # stable hash (not Python's salted hash()) so embeddings are reproducible
        # across process restarts
        digest = hashlib.md5(ngram.encode("utf-8")).hexdigest()
        return int(digest, 16) % self.dim

    def embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for gram in self._ngrams(text):
            idx = self._hash_to_index(gram)
            vector[idx] += 1.0

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.embed(t) for t in texts])
