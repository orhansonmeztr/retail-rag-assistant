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
from .telemetry import GEN_AI_SYSTEM, record_usage, span

log = logging.getLogger(__name__)

CATEGORIES = ["ayakkabi", "mont", "elbise", "gomlek", "canta", "pantolon", "kozmetik", "aksesuar"]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Search the product catalog. Use for product recommendations and "
            "for any detail of a specific product: sizes, colours, materials, prices, fit, "
            "care and packaging (e.g. whether it ships in a gift box).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language query in the customer's language, "
                        "with correct Turkish characters (ç ğ ı ö ş ü)",
                    },
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
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language query with correct Turkish characters",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
]

CITATION_RE = re.compile(r"\[([a-z0-9][a-z0-9\-]*)\]")

REFUSALS = {
    "tr": "Bu isteğe yardımcı olamıyorum. Ürünler, siparişler, iade, kargo veya ödeme "
    "hakkında bir sorunuz varsa memnuniyetle yardımcı olurum.",
    "en": "I can't help with that request. I'm happy to help with products, orders, "
    "returns, shipping or payments.",
}
_TR_HINT = re.compile(r"[çğıöşüÇĞİÖŞÜ]|(?i:\b(ve|bir|mi|mı|için|ne|bana|nasıl)\b)")


def guess_language(text: str) -> str:
    return "tr" if _TR_HINT.search(text) else "en"


def is_content_filter_error(exc: Exception) -> bool:
    """Azure OpenAI rejects prompts/completions flagged by its content filter with HTTP 400."""
    return getattr(exc, "code", None) == "content_filter" or (
        getattr(exc, "status_code", None) == 400 and "content management policy" in str(exc)
    )


GUARD_PROMPT = """You are a fact-checking filter for a store's customer assistant.
You receive SOURCES (search results and earlier conversation) and a DRAFT answer.
Delete only sentences or clauses that state a fact (price, fee, discount, promotion, policy,
delivery time, product property) which SOURCES do not contain or which contradicts SOURCES.
Do NOT delete a product just because it fits the question only partly (e.g. a unisex item
for a men's request): if its details are in SOURCES, keep it. Keep everything else exactly
as written: wording, language, formatting and [id] citations; if you delete something, make
sure the remaining text still reads correctly. Never add information.
If nothing must be removed, return DRAFT unchanged. Output only the final answer text."""


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
        answer_guard: bool = False,
        shield=None,
    ):
        self.chat_client = chat_client
        self.deployment = deployment
        self.retriever = retriever
        self.system_prompt = system_prompt
        self.prompt_version = prompt_version
        self.max_steps = max_steps
        self.top_k = top_k
        self.answer_guard = answer_guard
        self.shield = shield  # optional PromptShield-like object

    def _chat(self, operation: str = "chat", **kwargs):
        """One chat completion inside a GenAI span (model, tokens, finish reason)."""
        with span(
            f"{operation} {self.deployment}",
            **{
                "gen_ai.operation.name": "chat",
                "gen_ai.system": GEN_AI_SYSTEM,
                "gen_ai.request.model": self.deployment,
                "retail_rag.step": operation,
            },
        ) as s:
            resp = self.chat_client.chat.completions.create(model=self.deployment, **kwargs)
            record_usage(s, resp.usage)
            if getattr(resp.choices[0], "finish_reason", None):
                s.set_attribute("gen_ai.response.finish_reasons", [resp.choices[0].finish_reason])
            return resp

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

    def _guard(
        self, answer: str, docs: list[Document], history: list[ChatTurn], usage: Usage
    ) -> str:
        """Second line of defence: drop claims the sources don't support (e.g. a sentence the
        user dictated). Prompt rules alone did not stop that in evaluation (v1.2.0/v1.3.0)."""
        sources = "\n---\n".join(f"[{d.id}] {d.content}" for d in docs)
        earlier = "\n".join(f"{t.role}: {t.content}" for t in history)
        resp = self._chat(
            "answer_guard",
            temperature=0,
            messages=[
                {"role": "system", "content": GUARD_PROMPT},
                {
                    "role": "user",
                    "content": f"SOURCES:\n{sources or '-'}\n{earlier}\n\nDRAFT:\n{answer}",
                },
            ],
        )
        usage.add(resp.usage)
        return (resp.choices[0].message.content or answer).strip()

    @staticmethod
    def _dropped_products(before: str, after: str, seen: dict[str, Document]) -> set[str]:
        """Sourced products whose [id] or name was in the draft but is gone after the edit.
        Ids and names are compared separately: an edit can delete the sentence about a
        product yet leave its [id] behind in a citation group."""
        products = [d for d in seen.values() if d.source == "product"]

        def cited(text: str) -> set[str]:
            return set(CITATION_RE.findall(text)) & {d.id for d in products}

        def named(text: str) -> set[str]:
            low = text.casefold()
            # title is "<brand> <name>"; match on the full title or the product name alone
            return {
                d.id
                for d in products
                if d.title.casefold() in low or d.title.split(" ", 1)[-1].casefold() in low
            }

        return (cited(before) - cited(after)) | (named(before) - named(after))

    # ---- main loop ---------------------------------------------------------------------
    def ask(self, question: str, history: list[ChatTurn] | None = None) -> AgentAnswer:
        with span(
            "invoke_agent retail-assistant",
            **{
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.system": GEN_AI_SYSTEM,
                "gen_ai.request.model": self.deployment,
                "retail_rag.prompt_version": self.prompt_version,
                "retail_rag.question_chars": len(question),
                "retail_rag.history_turns": len(history or []),
            },
        ) as s:
            ans = self._ask(question, history)
            s.set_attribute("gen_ai.usage.input_tokens", ans.usage.prompt_tokens)
            s.set_attribute("gen_ai.usage.output_tokens", ans.usage.completion_tokens)
            s.set_attribute("retail_rag.tool_calls", len(ans.tool_calls))
            s.set_attribute("retail_rag.citations", len(ans.citations))
            s.set_attribute("retail_rag.blocked", ans.blocked)
            s.set_attribute("retail_rag.guard_edited", ans.guard_edited)
            return ans

    def _ask(self, question: str, history: list[ChatTurn] | None = None) -> AgentAnswer:
        started = time.perf_counter()
        messages: list[dict] = [{"role": "system", "content": self.system_prompt}]
        messages += [t.model_dump() for t in (history or [])]
        messages.append({"role": "user", "content": question})

        usage = Usage()
        traces: list[ToolCallTrace] = []
        seen: dict[str, Document] = {}
        answer = ""
        blocked = False
        blocked_by = None

        if self.shield is not None:
            with span("prompt_shield") as shield_span:
                try:
                    attack = self.shield.attack_detected(question)
                except Exception as exc:  # fail open: availability over a second safety net
                    log.warning("prompt shield unavailable: %s", exc)
                    attack = False
                shield_span.set_attribute("retail_rag.attack_detected", attack)
            if attack:
                log.warning("prompt attack detected by Prompt Shields")
                answer, blocked, blocked_by = (
                    REFUSALS[guess_language(question)],
                    True,
                    "prompt_shield",
                )

        for step in range(0 if blocked else self.max_steps + 1):
            force_answer = step == self.max_steps
            try:
                resp = self._chat(
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="none" if force_answer else "auto",
                    temperature=0,
                )
            except Exception as exc:
                if not is_content_filter_error(exc):
                    raise
                log.warning("blocked by Azure OpenAI content filter")
                answer, blocked = REFUSALS[guess_language(question)], True
                blocked_by = "content_filter"
                break
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
                with span(
                    f"execute_tool {tc.function.name}",
                    **{
                        "gen_ai.operation.name": "execute_tool",
                        "gen_ai.tool.name": tc.function.name,
                    },
                ) as tool_span:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                        docs = self._run_tool(tc.function.name, args)
                        content = self._tool_payload(docs)
                    except Exception as exc:  # surface tool errors to the model, don't crash
                        log.warning("tool %s failed: %s", tc.function.name, exc)
                        tool_span.set_attribute("error.type", type(exc).__name__)
                        args, docs, content = {}, [], json.dumps({"error": str(exc)})
                    tool_span.set_attribute("retail_rag.tool.result_count", len(docs))
                seen.update({d.id: d for d in docs})
                traces.append(
                    ToolCallTrace(
                        name=tc.function.name, arguments=args, result_ids=[d.id for d in docs]
                    )
                )
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})

        guarded = False
        if self.answer_guard and not blocked and seen and answer:
            try:
                checked = self._guard(answer, list(seen.values()), history or [], usage)
                if dropped := self._dropped_products(answer, checked, seen):
                    # products backed by sources must survive; the guard over-reached
                    log.warning("answer guard edit rejected, it removed %s", dropped)
                else:
                    guarded = checked != answer
                    answer = checked
            except Exception as exc:  # the guard must never take the answer down with it
                if not is_content_filter_error(exc):
                    log.warning("answer guard failed: %s", exc)

        cited_ids = list(dict.fromkeys(CITATION_RE.findall(answer)))
        citations = [
            Citation(id=i, title=seen[i].title, source=seen[i].source, url=seen[i].url)
            for i in cited_ids
            if i in seen  # drop hallucinated ids that were never retrieved
        ]
        latency_ms = int((time.perf_counter() - started) * 1000)
        log.info(
            "answered prompt_version=%s blocked=%s tools=%d tokens_in=%d tokens_out=%d "
            "latency_ms=%d",
            self.prompt_version,
            blocked,
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
            blocked=blocked,
            blocked_by=blocked_by,
            guard_edited=guarded,
            context_documents=list(seen.values()),
        )
