from retail_rag.models import Document, SearchFilter
from retail_rag.search_backend import AzureSearchBackend, reciprocal_rank_fusion


def test_odata_filter_rendering_and_escaping():
    f = SearchFilter(source="product", category="o'brien", gender="kadin", max_price=2000)
    assert f.to_odata() == (
        "source eq 'product' and category eq 'o''brien' and "
        "(gender eq 'kadin' or gender eq 'unisex') and price le 2000.0"
    )
    assert SearchFilter().to_odata() is None


def test_filter_matches_unisex_and_price():
    doc = Document(id="x", source="product", title="t", content="c", gender="unisex", price=100)
    assert SearchFilter(gender="kadin", max_price=150).matches(doc)
    assert not SearchFilter(max_price=50).matches(doc)
    assert not SearchFilter(source="faq").matches(doc)


def test_rrf_prefers_documents_ranked_well_in_both_lists():
    scores = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]])
    assert max(scores, key=scores.get) in {"a", "b"}
    assert scores["d"] < scores["a"]


def test_memory_backend_respects_filters(retriever):
    results = retriever.retrieve("mont", SearchFilter(source="product", max_price=2000))
    assert results
    assert all(r.document.price <= 2000 for r in results)


def test_memory_backend_finds_relevant_faq(retriever):
    results = retriever.retrieve("hediye kartı geçerlilik süresi", SearchFilter(source="faq"))
    assert results[0].document.id == "faq-hediye-karti-0"


def test_azure_index_schema_without_network():
    backend = AzureSearchBackend.__new__(AzureSearchBackend)
    backend.index_name, backend.dimensions = "test", 1536
    index = backend.build_index()
    names = {f.name for f in index.fields}
    assert {"id", "content", "content_vector", "price"} <= names
    vec = next(f for f in index.fields if f.name == "content_vector")
    assert vec.vector_search_dimensions == 1536


def test_base_endpoint_strips_foundry_path():
    from retail_rag.clients import base_endpoint

    url = "https://x-resource.services.ai.azure.com/openai/v1/responses"
    assert base_endpoint(url) == "https://x-resource.services.ai.azure.com/"
    assert base_endpoint("https://y.openai.azure.com") == "https://y.openai.azure.com/"


def test_openai_client_gets_timeout_and_retries():
    from retail_rag.clients import make_openai_client
    from retail_rag.config import Settings

    s = Settings(
        azure_openai_endpoint="https://x.openai.azure.com/abc",
        azure_openai_api_key="k",
        openai_timeout_s=12,
        openai_max_retries=5,
    )
    c = make_openai_client(s)
    assert c.max_retries == 5 and c.timeout == 12
    assert str(c.base_url).startswith("https://x.openai.azure.com/openai")
