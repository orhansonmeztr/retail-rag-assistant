"""Incremental index sync: re-embed only changed documents, delete the ones that disappeared.

Each indexed document stores a hash of its content. A catalog file is treated as the full
snapshot of one source ("product" or "faq"), so comparing hashes tells us exactly what to
add, update or delete, and an unchanged upload costs no embedding calls.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

from .documents import parse_faq, parse_products
from .models import Document, Source
from .retriever import Retriever

log = logging.getLogger(__name__)

CATALOG_FILES: dict[str, Source] = {"products.jsonl": "product", "faq.md": "faq"}


def content_hash(doc: Document) -> str:
    payload = doc.model_dump_json(exclude={"content_hash"})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class SyncReport:
    source: str
    added: int = 0
    updated: int = 0
    deleted: int = 0
    unchanged: int = 0


def sync_source(retriever: Retriever, source: Source, docs: list[Document]) -> SyncReport:
    backend = retriever.backend
    if any(d.source != source for d in docs):
        raise ValueError(f"all documents must have source={source!r}")
    existing = backend.list_hashes(source)
    current = {d.id: d.model_copy(update={"content_hash": content_hash(d)}) for d in docs}

    changed = [d for i, d in current.items() if existing.get(i) != d.content_hash]
    stale = sorted(set(existing) - set(current))
    if changed:
        retriever.index(changed)
    if stale:
        backend.delete(stale)

    report = SyncReport(
        source=source,
        added=sum(1 for d in changed if d.id not in existing),
        updated=sum(1 for d in changed if d.id in existing),
        deleted=len(stale),
        unchanged=len(current) - len(changed),
    )
    log.info("sync %s", report)
    return report


def handle_catalog_blob(name: str, text: str, retriever: Retriever) -> SyncReport:
    """Entry point for the blob trigger: parse the uploaded file and sync its source."""
    source = CATALOG_FILES.get(name)
    if source is None:
        expected = ", ".join(CATALOG_FILES)
        raise ValueError(f"unsupported catalog file {name!r}; expected one of: {expected}")
    docs = parse_products(text) if source == "product" else parse_faq(text)
    return sync_source(retriever, source, docs)


def build_sync_target(settings) -> Retriever:
    from .clients import make_openai_client, search_credential
    from .embeddings import AzureOpenAIEmbedder
    from .search_backend import AzureSearchBackend

    backend = AzureSearchBackend(
        settings.azure_search_endpoint,
        settings.azure_search_index,
        search_credential(settings),
        settings.embedding_dimensions,
        timeout_s=max(settings.search_timeout_s, 30.0),  # uploads are slower than queries
        max_retries=settings.search_max_retries,
    )
    backend.ensure_index()
    embedder = AzureOpenAIEmbedder(
        make_openai_client(settings),
        settings.azure_openai_embedding_deployment,
        settings.embedding_dimensions,
    )
    return Retriever(embedder, backend)
