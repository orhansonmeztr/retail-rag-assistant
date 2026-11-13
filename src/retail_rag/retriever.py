from __future__ import annotations

from .embeddings import Embedder
from .models import Document, SearchFilter, SearchResult
from .search_backend import SearchBackend


class Retriever:
    def __init__(self, embedder: Embedder, backend: SearchBackend, top_k: int = 5):
        self.embedder = embedder
        self.backend = backend
        self.top_k = top_k

    def index(self, docs: list[Document]) -> int:
        vectors = self.embedder.embed([f"{d.title}\n{d.content}" for d in docs])
        self.backend.upsert(docs, vectors)
        return len(docs)

    def retrieve(
        self, query: str, filter: SearchFilter | None = None, top_k: int | None = None
    ) -> list[SearchResult]:
        [vector] = self.embedder.embed([query])
        return self.backend.search(query, vector, top_k=top_k or self.top_k, filter=filter)
