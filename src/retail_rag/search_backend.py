"""Search backends behind one interface: Azure AI Search (prod) and in-memory (offline/tests)."""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

from .embeddings import tokenize
from .models import Document, SearchFilter, SearchResult

log = logging.getLogger(__name__)

VECTOR_FIELD = "content_vector"
SEMANTIC_CONFIG = "default"


class SearchBackend(Protocol):
    def upsert(self, docs: list[Document], vectors: list[list[float]]) -> None: ...

    def search(
        self,
        query: str,
        vector: list[float],
        top_k: int = 5,
        filter: SearchFilter | None = None,
    ) -> list[SearchResult]: ...

    def list_hashes(self, source: str) -> dict[str, str | None]:
        """id -> content_hash of every indexed document of one source."""
        ...

    def delete(self, ids: list[str]) -> None: ...


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Same fusion Azure AI Search uses for hybrid queries."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


class InMemoryBackend:
    """Hybrid (keyword + vector, RRF-fused) search in process memory."""

    def __init__(self) -> None:
        self._docs: dict[str, Document] = {}
        self._vectors: dict[str, np.ndarray] = {}

    def upsert(self, docs: list[Document], vectors: list[list[float]]) -> None:
        for doc, vec in zip(docs, vectors, strict=True):
            self._docs[doc.id] = doc
            v = np.asarray(vec, dtype=np.float32)
            self._vectors[doc.id] = v / (np.linalg.norm(v) or 1.0)

    def list_hashes(self, source: str) -> dict[str, str | None]:
        return {i: d.content_hash for i, d in self._docs.items() if d.source == source}

    def delete(self, ids: list[str]) -> None:
        for i in ids:
            self._docs.pop(i, None)
            self._vectors.pop(i, None)

    def search(self, query, vector, top_k=5, filter=None) -> list[SearchResult]:
        candidates = [d for d in self._docs.values() if filter is None or filter.matches(d)]
        if not candidates:
            return []
        q = np.asarray(vector, dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        by_vector = sorted(candidates, key=lambda d: -float(self._vectors[d.id] @ q))

        q_tokens = set(tokenize(query))

        def keyword_score(d: Document) -> float:
            toks = tokenize(f"{d.title} {d.content}")
            return sum(1 for t in toks if t in q_tokens) / (1 + len(toks)) ** 0.5

        by_keyword = [
            d for d in sorted(candidates, key=lambda d: -keyword_score(d)) if keyword_score(d) > 0
        ]

        fused = reciprocal_rank_fusion(
            [[d.id for d in by_vector[: top_k * 4]], [d.id for d in by_keyword[: top_k * 4]]]
        )
        ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]
        return [SearchResult(document=self._docs[i], score=s) for i, s in ranked]


class AzureSearchBackend:
    """Azure AI Search: hybrid BM25 + HNSW vector query, optional semantic re-ranking."""

    def __init__(
        self,
        endpoint: str,
        index_name: str,
        credential,
        dimensions: int = 1536,
        use_semantic_ranker: bool = False,
        timeout_s: float = 10.0,
        max_retries: int = 3,
    ) -> None:
        from azure.search.documents import SearchClient
        from azure.search.documents.indexes import SearchIndexClient

        self.index_name = index_name
        self.dimensions = dimensions
        self.use_semantic_ranker = use_semantic_ranker
        # azure-core defaults are a 300 s read timeout; a chat request must fail much sooner
        transport = {
            "connection_timeout": min(timeout_s, 5.0),
            "read_timeout": timeout_s,
            "retry_total": max_retries,
            "retry_backoff_factor": 0.5,
        }
        self.index_client = SearchIndexClient(endpoint=endpoint, credential=credential, **transport)
        self.client = SearchClient(
            endpoint=endpoint, index_name=index_name, credential=credential, **transport
        )

    def build_index(self):
        from azure.search.documents.indexes.models import (
            HnswAlgorithmConfiguration,
            SearchableField,
            SearchField,
            SearchFieldDataType,
            SearchIndex,
            SemanticConfiguration,
            SemanticField,
            SemanticPrioritizedFields,
            SemanticSearch,
            SimpleField,
            VectorSearch,
            VectorSearchProfile,
        )

        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
            SimpleField(
                name="source", type=SearchFieldDataType.String, filterable=True, facetable=True
            ),
            SearchableField(
                name="title", type=SearchFieldDataType.String, analyzer_name="tr.microsoft"
            ),
            SearchableField(
                name="content", type=SearchFieldDataType.String, analyzer_name="tr.microsoft"
            ),
            SimpleField(
                name="category", type=SearchFieldDataType.String, filterable=True, facetable=True
            ),
            SimpleField(name="gender", type=SearchFieldDataType.String, filterable=True),
            SimpleField(
                name="price", type=SearchFieldDataType.Double, filterable=True, sortable=True
            ),
            SimpleField(name="url", type=SearchFieldDataType.String),
            SimpleField(name="content_hash", type=SearchFieldDataType.String, filterable=True),
            SearchField(
                name=VECTOR_FIELD,
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=self.dimensions,
                vector_search_profile_name="hnsw-profile",
            ),
        ]
        vector_search = VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
            profiles=[
                VectorSearchProfile(name="hnsw-profile", algorithm_configuration_name="hnsw")
            ],
        )
        semantic = SemanticSearch(
            configurations=[
                SemanticConfiguration(
                    name=SEMANTIC_CONFIG,
                    prioritized_fields=SemanticPrioritizedFields(
                        title_field=SemanticField(field_name="title"),
                        content_fields=[SemanticField(field_name="content")],
                    ),
                )
            ]
        )
        return SearchIndex(
            name=self.index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_search=semantic,
        )

    def ensure_index(self) -> None:
        self.index_client.create_or_update_index(self.build_index())
        log.info("index %s ready", self.index_name)

    def upsert(self, docs: list[Document], vectors: list[list[float]]) -> None:
        payload = [{**d.model_dump(), VECTOR_FIELD: v} for d, v in zip(docs, vectors, strict=True)]
        for start in range(0, len(payload), 500):
            results = self.client.merge_or_upload_documents(payload[start : start + 500])
            failed = [r.key for r in results if not r.succeeded]
            if failed:
                raise RuntimeError(f"failed to index: {failed}")

    def list_hashes(self, source: str) -> dict[str, str | None]:
        results = self.client.search(
            search_text="*",
            filter=SearchFilter(source=source).to_odata(),
            select=["id", "content_hash"],
            top=1000,  # one page is plenty for this demo catalog
        )
        return {r["id"]: r.get("content_hash") for r in results}

    def delete(self, ids: list[str]) -> None:
        if ids:
            self.client.delete_documents([{"id": i} for i in ids])

    def search(self, query, vector, top_k=5, filter=None) -> list[SearchResult]:
        from azure.search.documents.models import VectorizedQuery

        kwargs = {}
        if self.use_semantic_ranker:
            kwargs = {"query_type": "semantic", "semantic_configuration_name": SEMANTIC_CONFIG}
        results = self.client.search(
            search_text=query,
            vector_queries=[
                VectorizedQuery(
                    vector=vector, k_nearest_neighbors=max(top_k, 10), fields=VECTOR_FIELD
                )
            ],
            filter=filter.to_odata() if filter else None,
            select=["id", "source", "title", "content", "category", "gender", "price", "url"],
            top=top_k,
            **kwargs,
        )
        out = []
        for r in results:
            score = r.get("@search.reranker_score") or r.get("@search.score") or 0.0
            doc = Document(**{k: r.get(k) for k in Document.model_fields})
            out.append(SearchResult(document=doc, score=float(score)))
        return out
