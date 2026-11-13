"""FastAPI service. Its OpenAPI spec (/openapi.json) can be imported into Copilot Studio
or Power Automate as a custom connector."""

from __future__ import annotations

import hashlib
import logging
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from . import __version__
from .agent import RetailAssistant
from .config import get_settings
from .factory import build_assistant
from .models import AgentAnswer, ChatRequest
from .security import RateLimiter, key_is_valid, parse_api_keys
from .telemetry import configure_telemetry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

configure_telemetry()  # no-op unless APPLICATIONINSIGHTS_CONNECTION_STRING is set

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if not get_api_keys():
        log.warning("API_KEYS is empty: /chat is open to anyone (fine locally, not in Azure)")
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


@lru_cache
def get_api_keys() -> tuple[str, ...]:
    return tuple(parse_api_keys(get_settings().api_keys))


@lru_cache
def get_rate_limiter() -> RateLimiter:
    return RateLimiter(get_settings().rate_limit_per_minute)


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def authorize(
    request: Request,
    api_key: Annotated[str | None, Security(api_key_header)],
    keys: Annotated[tuple[str, ...], Depends(get_api_keys)],
    limiter: Annotated[RateLimiter, Depends(get_rate_limiter)],
) -> str:
    """API-key check, then a per-client rate limit. Returns a non-secret client id."""
    if keys:
        if not key_is_valid(api_key, list(keys)):
            raise HTTPException(
                status_code=401,
                detail="invalid or missing API key",
                headers={"WWW-Authenticate": "ApiKey"},
            )
        client = "key:" + hashlib.sha256(api_key.encode()).hexdigest()[:12]
    else:
        client = "ip:" + (request.client.host if request.client else "unknown")
    if (retry_after := limiter.check(client)) is not None:
        raise HTTPException(
            status_code=429,
            detail="rate limit exceeded",
            headers={"Retry-After": str(retry_after)},
        )
    return client


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.post(
    "/chat",
    response_model=AgentAnswer,
    operation_id="AskAssistant",
    responses={
        401: {"description": "Missing or invalid API key"},
        429: {"description": "Rate limit exceeded"},
    },
)
def chat(
    req: ChatRequest,
    assistant: Annotated[RetailAssistant, Depends(get_assistant)],
    _client: Annotated[str, Depends(authorize)],
) -> AgentAnswer:
    try:
        return assistant.ask(req.question, req.history)
    except Exception as exc:
        log.exception("chat failed")
        raise HTTPException(status_code=502, detail="upstream model or search error") from exc
