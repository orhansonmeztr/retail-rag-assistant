import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from retail_rag.documents import load_corpus
from retail_rag.embeddings import HashingEmbedder
from retail_rag.retriever import Retriever
from retail_rag.search_backend import InMemoryBackend

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def corpus():
    return load_corpus(ROOT / "data")


@pytest.fixture(scope="session")
def retriever(corpus):
    r = Retriever(HashingEmbedder(), InMemoryBackend(), top_k=5)
    r.index(corpus)
    return r


def tool_call(call_id: str, name: str, args: dict):
    return SimpleNamespace(
        id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(args))
    )


def completion(content=None, tool_calls=None, prompt_tokens=10, completion_tokens=5):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg)],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


def stream(content=None, tool_calls=None, finish=None, prompt_tokens=10, completion_tokens=5):
    """Chunks shaped like Azure OpenAI streaming: a filter-results chunk without choices,
    content split into pieces, tool-call deltas split across chunks, then a usage chunk."""
    chunks = [SimpleNamespace(choices=[], usage=None)]

    def chunk(delta, finish_reason=None):
        return SimpleNamespace(
            choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)], usage=None
        )

    for piece in (content or "").split(" "):
        if piece:
            chunks.append(chunk(SimpleNamespace(content=piece + " ", tool_calls=None)))
    for i, (call_id, name, args) in enumerate(tool_calls or []):
        raw = json.dumps(args)
        first = SimpleNamespace(
            index=i, id=call_id, function=SimpleNamespace(name=name, arguments=raw[:5])
        )
        rest = SimpleNamespace(
            index=i, id=None, function=SimpleNamespace(name=None, arguments=raw[5:])
        )
        chunks.append(chunk(SimpleNamespace(content=None, tool_calls=[first])))
        chunks.append(chunk(SimpleNamespace(content=None, tool_calls=[rest])))
    chunks.append(
        chunk(
            SimpleNamespace(content=None, tool_calls=None),
            finish or ("tool_calls" if tool_calls else "stop"),
        )
    )
    chunks.append(
        SimpleNamespace(
            choices=[],
            usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
        )
    )
    return chunks


class ScriptedChatClient:
    """Fake Azure OpenAI client that replays scripted completions and records requests."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        resp = self._responses.pop(0)
        return iter(resp) if kwargs.get("stream") else resp
