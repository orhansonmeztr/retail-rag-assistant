from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Source = Literal["product", "faq"]


class Document(BaseModel):
    """A retrievable chunk: either one product or one FAQ section chunk."""

    id: str
    source: Source
    title: str
    content: str
    category: str | None = None
    gender: str | None = None
    price: float | None = None
    url: str | None = None
    # set by incremental sync; lets an upload skip documents that did not change
    content_hash: str | None = None


class SearchResult(BaseModel):
    document: Document
    score: float


Gender = Literal["kadin", "erkek", "unisex"]


class SearchFilter(BaseModel):
    """Structured filters the agent can pass; rendered to OData for Azure AI Search.

    `gender` selects products for that gender and, unless `include_unisex` is False, unisex
    products too: someone shopping for men can wear a unisex jacket. Both the in-memory
    filter and the OData filter are derived from `allowed_genders()`, so they cannot drift.
    """

    source: Source | None = None
    category: str | None = None
    gender: Gender | None = None
    include_unisex: bool = True
    max_price: float | None = None

    def allowed_genders(self) -> list[str] | None:
        if self.gender is None:
            return None
        if self.include_unisex and self.gender != "unisex":
            return [self.gender, "unisex"]
        return [self.gender]

    def matches(self, doc: Document) -> bool:
        if self.source and doc.source != self.source:
            return False
        if self.category and doc.category != self.category:
            return False
        genders = self.allowed_genders()
        if genders is not None and doc.gender not in genders:
            return False
        return not (
            self.max_price is not None and (doc.price is None or doc.price > self.max_price)
        )

    def to_odata(self) -> str | None:
        def q(v: str) -> str:
            return "'" + v.replace("'", "''") + "'"

        parts: list[str] = []
        if self.source:
            parts.append(f"source eq {q(self.source)}")
        if self.category:
            parts.append(f"category eq {q(self.category)}")
        if genders := self.allowed_genders():
            ors = " or ".join(f"gender eq {q(g)}" for g in genders)
            parts.append(f"({ors})" if len(genders) > 1 else ors)
        if self.max_price is not None:
            parts.append(f"price le {float(self.max_price)}")
        return " and ".join(parts) or None


class ToolCallTrace(BaseModel):
    name: str
    arguments: dict
    result_ids: list[str]


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, other) -> None:
        if other is None:
            return
        self.prompt_tokens += getattr(other, "prompt_tokens", 0) or 0
        self.completion_tokens += getattr(other, "completion_tokens", 0) or 0


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=20)


class Citation(BaseModel):
    id: str
    title: str
    source: Source
    url: str | None = None


class AgentAnswer(BaseModel):
    answer: str
    citations: list[Citation]
    tool_calls: list[ToolCallTrace]
    usage: Usage
    prompt_version: str
    latency_ms: int
    blocked: bool = False  # True when the request was refused before answering
    blocked_by: str | None = None  # "prompt_shield" or "content_filter"
    guard_edited: bool = False  # True when the answer guard removed unsupported claims
    # documents the tools returned; used by evaluation, not sent to API clients
    context_documents: list[Document] = Field(default_factory=list, exclude=True)
