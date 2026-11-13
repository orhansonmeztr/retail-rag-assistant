"""FastAPI service. Its OpenAPI spec (/openapi.json) can be imported into Copilot Studio
or Power Automate as a custom connector."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException

from . import __version__
from .agent import RetailAssistant
from .config import get_settings
from .factory import build_assistant
from .models import AgentAnswer, ChatRequest
from .telemetry import configure_telemetry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

configure_telemetry()  # no-op unless APPLICATIONINSIGHTS_CONNECTION_STRING is set

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if get_settings().warmup_on_startup:
        # Tracing showed the first request paying ~3 s for TLS/connection setup to Azure
        # OpenAI and AI Search (later requests: embed ~0.3 s, search ~0.5 s). One dummy
        # retrieval at startup moves that cost out of the first customer's request.
        try:
            get_assistant().retriever.retrieve("warmup", top_k=1)
            log.info("warm-up done")
        except Exception as exc:  # a failed warm-up must not stop the service
            log.warning("warm-up failed: %s", exc)
    yield


app = FastAPI(
    lifespan=lifespan,
    title="Retail RAG Assistant",
    version=__version__,
    description="Grounded Q&A over a product catalog and customer-service FAQ.",
)


@lru_cache
def get_assistant() -> RetailAssistant:
    return build_assistant(get_settings())


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.post("/chat", response_model=AgentAnswer, operation_id="AskAssistant")
def chat(
    req: ChatRequest, assistant: Annotated[RetailAssistant, Depends(get_assistant)]
) -> AgentAnswer:
    try:
        return assistant.ask(req.question, req.history)
    except Exception as exc:
        logging.getLogger(__name__).exception("chat failed")
        raise HTTPException(status_code=502, detail="upstream model or search error") from exc
