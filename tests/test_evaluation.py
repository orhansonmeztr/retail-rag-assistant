from conftest import ROOT

from retail_rag.evaluation import (
    evaluate_retrieval,
    load_golden,
    recall_at_k,
    reciprocal_rank,
)


def test_metrics():
    assert recall_at_k(["a", "b", "c"], ["c"], k=2) == 0.0
    assert recall_at_k(["a", "b", "c"], ["c"], k=3) == 1.0
    assert reciprocal_rank(["a", "b"], ["b"]) == 0.5


def test_golden_ids_exist_in_corpus(corpus):
    ids = {d.id for d in corpus}
    for ex in load_golden(ROOT / "eval" / "golden.jsonl"):
        assert set(ex.expected_ids) <= ids, ex.question


def test_offline_retrieval_baseline(retriever):
    """Quality gate: even the hashing baseline must clear this bar. Real embeddings do better."""
    report = evaluate_retrieval(retriever, load_golden(ROOT / "eval" / "golden.jsonl"), k=5)
    assert report.recall_at_k >= 0.7, report.misses


def test_summarize_citation_metrics():
    from retail_rag.evaluation import summarize

    base = {"hit": True, "groundedness": 5, "relevance": 4, "latency_ms": 10, "tokens": 100}
    rows = [
        {**base, "has_citation": True, "expected_cited": True},
        {**base, "has_citation": False, "expected_cited": False},
    ]
    s = summarize("1.0.0", rows)
    assert s["citation_rate"] == 0.5 and s["expected_cited_rate"] == 0.5
