from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

from .documents import _TR_MAP


class Embedder(Protocol):
    dimensions: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class AzureOpenAIEmbedder:
    """Batched embeddings from an Azure OpenAI embedding deployment."""

    def __init__(self, client, deployment: str, dimensions: int = 1536, batch_size: int = 64):
        self.client = client
        self.deployment = deployment
        self.dimensions = dimensions
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            resp = self.client.embeddings.create(
                model=self.deployment, input=batch, dimensions=self.dimensions
            )
            vectors.extend(item.embedding for item in sorted(resp.data, key=lambda d: d.index))
        return vectors


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.translate(_TR_MAP).lower())


class HashingEmbedder:
    """Deterministic, dependency-free embedder (word + char n-gram feature hashing).

    Not semantic like a real model, but char n-grams cope reasonably with Turkish suffixes
    (ayakkabı / ayakkabılar). Used for offline demos and unit tests.
    """

    def __init__(self, dimensions: int = 512, ngram: int = 4):
        self.dimensions = dimensions
        self.ngram = ngram

    def _features(self, text: str) -> list[str]:
        feats: list[str] = []
        for tok in tokenize(text):
            feats.append(f"w:{tok}")
            padded = f"#{tok}#"
            feats.extend(
                f"c:{padded[i : i + self.ngram]}"
                for i in range(max(1, len(padded) - self.ngram + 1))
            )
        return feats

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dimensions
            for f in self._features(text):
                h = int.from_bytes(hashlib.md5(f.encode()).digest()[:8], "little")
                vec[h % self.dimensions] += 1.0 if (h >> 63) == 0 else -1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out
