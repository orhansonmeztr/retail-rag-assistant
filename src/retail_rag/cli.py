"""Command line: `retail-rag ingest | ask | eval-retrieval | eval`."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from .config import get_settings
from .documents import load_corpus


def cmd_ingest(_args) -> None:
    from .clients import make_openai_client, search_credential
    from .embeddings import AzureOpenAIEmbedder
    from .retriever import Retriever
    from .search_backend import AzureSearchBackend

    s = get_settings()
    docs = load_corpus(s.data_dir)
    backend = AzureSearchBackend(
        s.azure_search_endpoint, s.azure_search_index, search_credential(s), s.embedding_dimensions
    )
    backend.ensure_index()
    embedder = AzureOpenAIEmbedder(
        make_openai_client(s), s.azure_openai_embedding_deployment, s.embedding_dimensions
    )
    n = Retriever(embedder, backend).index(docs)
    print(f"indexed {n} documents into '{s.azure_search_index}'")


def cmd_ask(args) -> None:
    from .factory import build_assistant

    ans = build_assistant(get_settings()).ask(args.question)
    print(ans.answer)
    print("\nSources:", ", ".join(c.id for c in ans.citations) or "-")
    print(
        f"[prompt v{ans.prompt_version} | {ans.latency_ms} ms | "
        f"{ans.usage.prompt_tokens}+{ans.usage.completion_tokens} tokens]"
    )


def cmd_eval_retrieval(args) -> None:
    from .evaluation import evaluate_retrieval, load_golden
    from .factory import build_retriever

    s = get_settings()
    if args.offline:
        s = s.model_copy(update={"search_backend": "memory"})
    retriever = build_retriever(s)
    report = evaluate_retrieval(retriever, load_golden(s.golden_path), k=args.k)
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
    if args.min_recall and report.recall_at_k < args.min_recall:
        sys.exit(f"recall@{args.k}={report.recall_at_k:.2f} < {args.min_recall}")


def cmd_eval(args) -> None:
    from .clients import make_openai_client
    from .evaluation import evaluate_answers, load_golden
    from .factory import build_assistant

    s = get_settings()
    report = evaluate_answers(
        build_assistant(s),
        load_golden(s.golden_path),
        make_openai_client(s),
        args.judge_deployment or s.azure_openai_chat_deployment,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {k: v for k, v in report.items() if k != "rows"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.WARNING)
    p = argparse.ArgumentParser(prog="retail-rag")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ingest", help="create the Azure AI Search index and upload documents")

    a = sub.add_parser("ask", help="ask the assistant a question")
    a.add_argument("question")

    r = sub.add_parser("eval-retrieval", help="recall@k / MRR on the golden set")
    r.add_argument("-k", type=int, default=5)
    r.add_argument(
        "--offline",
        action="store_true",
        help="use the in-memory index + hashing embedder (no Azure needed)",
    )
    r.add_argument(
        "--min-recall",
        type=float,
        default=None,
        help="exit non-zero if recall@k falls below this (quality gate)",
    )

    e = sub.add_parser("eval", help="end-to-end answer evaluation with an LLM judge")
    e.add_argument("--out", default="reports/eval.json")
    e.add_argument("--judge-deployment", default=None)

    args = p.parse_args(argv)
    {"ingest": cmd_ingest, "ask": cmd_ask, "eval-retrieval": cmd_eval_retrieval, "eval": cmd_eval}[
        args.cmd
    ](args)


if __name__ == "__main__":
    main()
