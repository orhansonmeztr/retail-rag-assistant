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


class ScriptedChatClient:
    """Fake Azure OpenAI client that replays scripted completions and records requests."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return self._responses.pop(0)
