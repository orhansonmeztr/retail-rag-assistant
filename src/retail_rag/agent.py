"""Tool-calling RAG agent: the LLM picks the knowledge source and filters to query."""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

import yaml

from .models import (
    AgentAnswer,
    ChatTurn,
    Citation,
    Document,
    SearchFilter,
    ToolCallTrace,
    Usage,
)
from .retriever import Retriever

log = logging.getLogger(__name__)

CATEGORIES = ["ayakkabi", "mont", "elbise", "gomlek", "canta", "pantolon", "kozmetik", "aksesuar"]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Search the product catalog. Use for product recommendations, "
            "availability of sizes, colours, materials and prices.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural-language search query"},
                    "category": {"type": "string", "enum": CATEGORIES},
                    "gender": {"type": "string", "enum": ["kadin", "erkek", "unisex"]},
                    "max_price": {"type": "number", "description": "Maximum price in TL"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_policies",
            "description": "Search customer-service policies: returns, exchanges, shipping, "
            "click & collect, cancellation, payment, instalments, loyalty points, gift cards.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
]

CITATION_RE = re.compile(r"\[([a-z0-9][a-z0-9\-]*)\]")


def load_prompt(path: Path) -> tuple[str, str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data["system"], str(data["version"])


class RetailAssistant:
    def __init__(
        self,
        chat_client,
        deployment: str,
        retriever: Retriever,
        system_prompt: str,
        prompt_version: str,
        max_steps: int = 4,
        top_k: int = 5,
    ):
        self.chat_client = chat_client
        self.deployment = deployment
        self.retriever = retriever
        self.system_prompt = system_prompt
        self.prompt_version = prompt_version
        self.max_steps = max_steps
        self.top_k = top_k

    # ---- tools -------------------------------------------------------------------------
    def _run_tool(self, name: str, args: dict) -> list[Document]:
        if name == "search_products":
            flt = SearchFilter(
                source="product",
                category=args.get("category"),
                gender=args.get("gender"),
                max_price=args.get("max_price"),
            )
        elif name == "search_policies":
            flt = SearchFilter(source="faq")
        else:
            raise ValueError(f"unknown tool {name}")
        return [r.document for r in self.retriever.retrieve(args["query"], flt, self.top_k)]

    @staticmethod
    def _tool_payload(docs: list[Document]) -> str:
        return json.dumps(
            [
                {"id": d.id, "title": d.title, "content": d.content, "price_tl": d.price}
                for d in docs
            ],
            ensure_ascii=False,
        )

    # ---- main loop ---------------------------------------------------------------------
    def ask(self, question: str, history: list[ChatTurn] | None = None) -> AgentAnswer:
        started = time.perf_counter()
        messages: list[dict] = [{"role": "system", "content": self.system_prompt}]
        messages += [t.model_dump() for t in (history or [])]
        messages.append({"role": "user", "content": question})

        usage = Usage()
        traces: list[ToolCallTrace] = []
        seen: dict[str, Document] = {}
        answer = ""

        for step in range(self.max_steps + 1):
            force_answer = step == self.max_steps
            resp = self.chat_client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                tools=TOOLS,
                tool_choice="none" if force_answer else "auto",
                temperature=0,
            )
            usage.add(resp.usage)
            msg = resp.choices[0].message
            if not msg.tool_calls:
                answer = msg.content or ""
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
            )
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                    docs = self._run_tool(tc.function.name, args)
                    content = self._tool_payload(docs)
                except Exception as exc:  # surface tool errors to the model, don't crash
                    log.warning("tool %s failed: %s", tc.function.name, exc)
                    args, docs, content = {}, [], json.dumps({"error": str(exc)})
                seen.update({d.id: d for d in docs})
                traces.append(
                    ToolCallTrace(
                        name=tc.function.name, arguments=args, result_ids=[d.id for d in docs]
                    )
                )
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})

        cited_ids = list(dict.fromkeys(CITATION_RE.findall(answer)))
        citations = [
            Citation(id=i, title=seen[i].title, source=seen[i].source, url=seen[i].url)
            for i in cited_ids
            if i in seen  # drop hallucinated ids that were never retrieved
        ]
        latency_ms = int((time.perf_counter() - started) * 1000)
        log.info(
            "answered prompt_version=%s tools=%d tokens_in=%d tokens_out=%d latency_ms=%d",
            self.prompt_version,
            len(traces),
            usage.prompt_tokens,
            usage.completion_tokens,
            latency_ms,
        )
        return AgentAnswer(
            answer=answer,
            citations=citations,
            tool_calls=traces,
            usage=usage,
            prompt_version=self.prompt_version,
            latency_ms=latency_ms,
            context_documents=list(seen.values()),
        )
