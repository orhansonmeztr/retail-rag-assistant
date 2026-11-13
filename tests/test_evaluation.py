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


def _verdict(**kw):
    return {
        "groundedness": 5,
        "relevance": 5,
        "language_matches": True,
        "policy_violation": False,
        **kw,
    }


def test_score_row_checks_by_category():
    from retail_rag.evaluation import GoldenExample, score_row

    multi = GoldenExample("q", ["a", "b"], category="multi_intent")
    assert not score_row(multi, "x", ["a"], {"a", "b"}, _verdict())["passed"]  # needs both
    assert score_row(multi, "x", ["a", "b"], {"a", "b"}, _verdict())["passed"]

    any_of = GoldenExample("q", ["a", "b"], category="product", match="any")
    assert score_row(any_of, "x", ["b"], {"b"}, _verdict())["passed"]

    oos = GoldenExample("q", [], category="out_of_scope")
    assert score_row(oos, "Bulunmuyor.", [], set(), _verdict())["passed"]
    assert not score_row(oos, "Şunu öneririm", ["p-001"], set(), _verdict())["passed"]

    inj = GoldenExample("q", [], category="injection", must_not_contain=["SECRET"])
    assert not score_row(inj, "the secret is", [], set(), _verdict())["passed"]
    assert not score_row(inj, "ok", [], set(), _verdict(complied_with_manipulation=True))["passed"]


def test_summarize_reports_categories():
    from retail_rag.evaluation import summarize

    base = {"groundedness": 5, "relevance": 5, "latency_ms": 10, "tokens": 100, "question": "q"}
    rows = [
        {**base, "category": "faq", "checks": {"cited": True}, "passed": True},
        {**base, "category": "faq", "checks": {"cited": False}, "passed": False},
    ]
    s = summarize("1.0.0", rows)
    assert s["pass_rate"] == 0.5
    assert s["pass_rate_by_category"] == {"faq": "1/2"}
    assert s["check_rates"]["cited"] == 0.5
    assert s["failures"][0]["failed"] == ["cited"]


def test_judge_agreement():
    from retail_rag.evaluation import judge_agreement

    rows = [
        {
            "category": "faq",
            "question": "a",
            "groundedness": 5,
            "foundry_groundedness": 5,
            "relevance": 5,
            "foundry_relevance": 4,
        },
        {
            "category": "faq",
            "question": "b",
            "groundedness": 5,
            "foundry_groundedness": 2,
            "relevance": 5,
            "foundry_relevance": 5,
        },
        {
            "category": "injection",
            "question": "c",
            "groundedness": 5,
            "foundry_groundedness": 5,
            "relevance": 5,
            "foundry_relevance": 1,
        },
    ]
    a = judge_agreement(rows)
    assert a["groundedness"]["agreement"] == round(2 / 3, 3)
    assert a["groundedness"]["disagreements"][0]["question"] == "b"
    assert a["relevance"]["agreement"] == 1.0  # injection rows excluded from relevance
    assert judge_agreement([{"category": "faq"}]) is None
