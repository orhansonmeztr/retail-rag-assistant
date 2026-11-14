"""Command line: `retail-rag ingest | ask | eval-retrieval | eval`."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from .config import get_settings


def cmd_ingest(_args) -> None:
    """Create/update the index schema, then sync both catalog files incrementally."""
    from .sync import CATALOG_FILES, build_sync_target, handle_catalog_blob

    s = get_settings()
    retriever = build_sync_target(s)
    for name in CATALOG_FILES:
        text = (s.data_dir / name).read_text(encoding="utf-8")
        print(handle_catalog_blob(name, text, retriever))


def cmd_sync(args) -> None:
    """Sync one catalog file, exactly what the Azure Function does on upload."""
    from .sync import build_sync_target, handle_catalog_blob

    path = Path(args.path)
    retriever = build_sync_target(get_settings())
    print(handle_catalog_blob(path.name, path.read_text(encoding="utf-8"), retriever))


def cmd_ask(args) -> None:
    import time

    from .factory import build_assistant

    s = get_settings()
    if args.no_guard:
        s = s.model_copy(update={"answer_guard": False})
    assistant = build_assistant(s)
    if not args.stream:
        ans = assistant.ask(args.question)
        print(ans.answer)
    else:
        started, first = time.perf_counter(), None
        for ev in assistant.ask_stream(args.question):
            if first is None:
                first = time.perf_counter() - started
            if ev["type"] == "token":
                print(ev["text"], end="", flush=True)
            elif ev["type"] in ("tool_start", "status", "blocked"):
                print(
                    f"[{ev['type']}: {ev.get('name') or ev.get('text') or ev.get('by')}]",
                    flush=True,
                )
            elif ev["type"] == "done":
                ans = ev["answer"]
                if assistant.answer_guard or ans.blocked:  # answer was not token-streamed
                    print(ans.answer)
        print(f"\n[first event after {first * 1000:.0f} ms]")
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
    from .clients import base_endpoint, make_openai_client
    from .evaluation import FoundryJudge, evaluate_answers, load_golden
    from .factory import build_assistant

    s = get_settings()
    judge_deployment = args.judge_deployment or s.judge_deployment or s.azure_openai_chat_deployment
    second = None
    if args.foundry_judge:
        second = FoundryJudge(
            base_endpoint(s.azure_openai_endpoint),
            s.azure_openai_api_key,
            judge_deployment,
            s.azure_openai_api_version,
        )
    report = evaluate_answers(
        build_assistant(s),
        load_golden(s.golden_path),
        make_openai_client(s),
        judge_deployment,
        workers=args.workers,
        second_judge=second,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {k: v for k, v in report.items() if k != "rows"}
    if args.min_pass_rate and report["pass_rate"] < args.min_pass_rate:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        sys.exit(f"pass_rate={report['pass_rate']:.2f} < {args.min_pass_rate}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.WARNING)
    p = argparse.ArgumentParser(prog="retail-rag")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ingest", help="create/update the index and sync all catalog files")

    sy = sub.add_parser("sync", help="incrementally sync one file (products.jsonl or faq.md)")
    sy.add_argument("path")

    a = sub.add_parser("ask", help="ask the assistant a question")
    a.add_argument("question")
    a.add_argument("--stream", action="store_true", help="print events/tokens as they arrive")
    a.add_argument("--no-guard", action="store_true", help="disable the answer guard")

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
    e.add_argument("--workers", type=int, default=4, help="questions evaluated in parallel")
    e.add_argument("--min-pass-rate", type=float, default=None)
    e.add_argument(
        "--foundry-judge",
        action="store_true",
        help="also score with azure-ai-evaluation and report judge agreement",
    )

    args = p.parse_args(argv)
    from .telemetry import configure_telemetry

    configure_telemetry(get_settings().applicationinsights_connection_string)
    {
        "ingest": cmd_ingest,
        "sync": cmd_sync,
        "ask": cmd_ask,
        "eval-retrieval": cmd_eval_retrieval,
        "eval": cmd_eval,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
