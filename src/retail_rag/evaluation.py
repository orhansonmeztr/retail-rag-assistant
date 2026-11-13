"""Offline evaluation: retrieval quality (recall@k, MRR) and answer quality (LLM-as-judge)."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from .agent import RetailAssistant
from .models import SearchFilter
from .retriever import Retriever


@dataclass
class GoldenExample:
    question: str
    expected_ids: list[str]
    filter: SearchFilter | None = None


def load_golden(path: Path) -> list[GoldenExample]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            flt = SearchFilter(**row["filter"]) if row.get("filter") else None
            out.append(GoldenExample(row["question"], row["expected_ids"], flt))
    return out


def recall_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    return len(set(retrieved[:k]) & set(expected)) / len(expected)


def reciprocal_rank(retrieved: list[str], expected: list[str]) -> float:
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in expected:
            return 1.0 / rank
    return 0.0


@dataclass
class RetrievalReport:
    k: int
    recall_at_k: float
    mrr: float
    misses: list[dict] = field(default_factory=list)


def evaluate_retrieval(retriever: Retriever, golden: list[GoldenExample], k: int = 5):
    recalls, rrs, misses = [], [], []
    for ex in golden:
        ids = [r.document.id for r in retriever.retrieve(ex.question, ex.filter, top_k=k)]
        r = recall_at_k(ids, ex.expected_ids, k)
        recalls.append(r)
        rrs.append(reciprocal_rank(ids, ex.expected_ids))
        if r < 1.0:
            misses.append({"question": ex.question, "expected": ex.expected_ids, "got": ids})
    return RetrievalReport(k, statistics.mean(recalls), statistics.mean(rrs), misses)


JUDGE_PROMPT = """You are grading a retail customer assistant.
Given the QUESTION, the retrieved CONTEXT and the ANSWER, return JSON:
{"groundedness": 1-5, "relevance": 1-5, "reason": "<one sentence>"}
groundedness: 5 = every claim is supported by CONTEXT, 1 = mostly unsupported.
relevance: 5 = fully answers the question, 1 = off-topic."""


def judge(client, deployment: str, question: str, context: str, answer: str) -> dict:
    resp = client.chat.completions.create(
        model=deployment,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {
                "role": "user",
                "content": f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}",
            },
        ],
    )
    return json.loads(resp.choices[0].message.content)


def evaluate_answers(
    assistant: RetailAssistant, golden: list[GoldenExample], judge_client, judge_deployment: str
) -> dict:
    rows = []
    for ex in golden:
        ans = assistant.ask(ex.question)
        retrieved = {i for t in ans.tool_calls for i in t.result_ids}
        context_docs = [r.document for r in assistant.retriever.retrieve(ex.question, ex.filter)]
        context = "\n---\n".join(f"[{d.id}] {d.content}" for d in context_docs)
        scores = judge(judge_client, judge_deployment, ex.question, context, ans.answer)
        rows.append(
            {
                "question": ex.question,
                "answer": ans.answer,
                "hit": bool(retrieved & set(ex.expected_ids)),
                "cited": [c.id for c in ans.citations],
                "latency_ms": ans.latency_ms,
                "tokens": ans.usage.prompt_tokens + ans.usage.completion_tokens,
                **scores,
            }
        )
    return {
        "prompt_version": assistant.prompt_version,
        "n": len(rows),
        "tool_hit_rate": statistics.mean(r["hit"] for r in rows),
        "groundedness": statistics.mean(r["groundedness"] for r in rows),
        "relevance": statistics.mean(r["relevance"] for r in rows),
        "p50_latency_ms": statistics.median(r["latency_ms"] for r in rows),
        "avg_tokens": statistics.mean(r["tokens"] for r in rows),
        "rows": rows,
    }
