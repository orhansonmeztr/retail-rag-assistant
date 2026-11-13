from __future__ import annotations

from .embeddings import Embedder
from .models import Document, SearchFilter, SearchResult
from .search_backend import SearchBackend
from .telemetry import span


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
        k = top_k or self.top_k
        with span(
            "retrieve",
            **{
                "retail_rag.backend": type(self.backend).__name__,
                "retail_rag.top_k": k,
                "retail_rag.filter": filter.to_odata() if filter else None,
            },
        ) as s:
            with span("embed_query"):
                [vector] = self.embedder.embed([query])
            results = self.backend.search(query, vector, top_k=k, filter=filter)
            s.set_attribute("retail_rag.result_count", len(results))
            s.set_attribute("retail_rag.result_ids", [r.document.id for r in results])
            return results
