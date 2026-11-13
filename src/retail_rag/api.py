"""FastAPI service. Its OpenAPI spec (/openapi.json) can be imported into Copilot Studio
or Power Automate as a custom connector."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException

from . import __version__
from .agent import RetailAssistant
from .config import get_settings
from .factory import build_assistant
from .models import AgentAnswer, ChatRequest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

app = FastAPI(
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
