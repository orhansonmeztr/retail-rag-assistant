import json

import pytest

from retail_rag.documents import parse_faq
from retail_rag.embeddings import HashingEmbedder
from retail_rag.retriever import Retriever
from retail_rag.search_backend import InMemoryBackend
from retail_rag.sync import handle_catalog_blob, sync_source


class CountingEmbedder(HashingEmbedder):
    def __init__(self):
        super().__init__()
        self.embedded = 0

    def embed(self, texts):
        self.embedded += len(texts)
        return super().embed(texts)


def product(pid, price=100.0, desc="açıklama"):
    return {
        "id": pid,
        "name": f"Ürün {pid}",
        "brand": "Marka",
        "category": "mont",
        "gender": "unisex",
        "price": price,
        "color": "siyah",
        "sizes": ["M"],
        "material": "pamuk",
        "description": desc,
    }


def jsonl(*rows):
    return "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)


@pytest.fixture
def target():
    emb = CountingEmbedder()
    return Retriever(emb, InMemoryBackend()), emb


def test_first_sync_adds_everything(target):
    retriever, emb = target
    report = handle_catalog_blob("products.jsonl", jsonl(product("a"), product("b")), retriever)
    assert (report.added, report.updated, report.deleted, report.unchanged) == (2, 0, 0, 0)
    assert emb.embedded == 2


def test_resync_only_embeds_changes_and_deletes_removed(target):
    retriever, emb = target
    handle_catalog_blob(
        "products.jsonl", jsonl(product("a"), product("b"), product("c")), retriever
    )
    emb.embedded = 0
    # a unchanged, b price changed, c removed, d new
    report = handle_catalog_blob(
        "products.jsonl", jsonl(product("a"), product("b", price=90), product("d")), retriever
    )
    assert (report.added, report.updated, report.deleted, report.unchanged) == (1, 1, 1, 1)
    assert emb.embedded == 2  # only b and d were re-embedded
    assert set(retriever.backend.list_hashes("product")) == {"a", "b", "d"}


def test_identical_upload_costs_no_embeddings(target):
    retriever, emb = target
    text = jsonl(product("a"))
    handle_catalog_blob("products.jsonl", text, retriever)
    emb.embedded = 0
    assert handle_catalog_blob("products.jsonl", text, retriever).unchanged == 1
    assert emb.embedded == 0


def test_faq_sync_does_not_touch_products(target):
    retriever, _ = target
    handle_catalog_blob("products.jsonl", jsonl(product("a")), retriever)
    handle_catalog_blob("faq.md", "# SSS\n\n## Kargo\n\nKargo ücretsizdir.\n", retriever)
    assert set(retriever.backend.list_hashes("product")) == {"a"}
    assert set(retriever.backend.list_hashes("faq")) == {"faq-kargo-0"}


def test_unknown_file_and_wrong_source_rejected(target):
    retriever, _ = target
    with pytest.raises(ValueError):
        handle_catalog_blob("prices.csv", "x", retriever)
    with pytest.raises(ValueError):
        sync_source(retriever, "product", parse_faq("## A\n\nb"))


def test_function_app_binding():
    """The Functions entry point loads and is bound to the catalog container via Event Grid."""
    import importlib.util
    import sys
    from pathlib import Path

    pytest.importorskip("azure.functions")
    path = Path(__file__).resolve().parents[1] / "functions" / "function_app.py"
    spec = importlib.util.spec_from_file_location("function_app", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["function_app"] = module
    spec.loader.exec_module(module)
    [fn] = module.app.get_functions()
    [binding] = fn.get_bindings()
    assert binding.type == "blobTrigger" and binding.path == "catalog/{name}"
    assert binding.connection == "CatalogStorage"
