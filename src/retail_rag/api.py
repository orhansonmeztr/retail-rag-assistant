"""FastAPI service. Its OpenAPI spec (/openapi.json) can be imported into Copilot Studio
or Power Automate as a custom connector."""

from __future__ import annotations

import hashlib
import json
import logging
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader

from . import __version__
from .agent import RetailAssistant
from .config import get_settings
from .factory import build_assistant
from .models import AgentAnswer, ChatRequest
from .security import RateLimiter, key_is_valid, parse_api_keys
from .telemetry import configure_telemetry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
# the Azure SDK logs every request's headers at INFO; keep only its warnings
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)

# no-op unless APPLICATIONINSIGHTS_CONNECTION_STRING is set (environment or .env)
configure_telemetry(get_settings().applicationinsights_connection_string)

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


def assistant_or_503() -> RetailAssistant:
    try:
        return get_assistant()
    except Exception as exc:  # missing endpoints/keys: say so instead of a bare 500
        log.error("assistant could not be built: %s", exc)
        raise HTTPException(status_code=503, detail="assistant not configured") from exc


@lru_cache
def get_api_keys() -> tuple[str, ...]:
    return tuple(parse_api_keys(get_settings().api_keys))


@lru_cache
def get_rate_limiter() -> RateLimiter:
    return RateLimiter(get_settings().rate_limit_per_minute)


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def upstream_error(exc: Exception) -> HTTPException:
    """Map a failure of Azure OpenAI / AI Search to a status that tells clients whether a
    retry makes sense: 504 timeout, 503 busy (after the SDK's own retries), 502 otherwise."""
    import openai
    from azure.core.exceptions import ServiceRequestTimeoutError, ServiceResponseTimeoutError

    if isinstance(
        exc, openai.APITimeoutError | ServiceRequestTimeoutError | ServiceResponseTimeoutError
    ):
        return HTTPException(status_code=504, detail="upstream timeout")
    if isinstance(exc, openai.RateLimitError):
        return HTTPException(
            status_code=503, detail="model is busy, retry later", headers={"Retry-After": "10"}
        )
    return HTTPException(status_code=502, detail="upstream model or search error")


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
    _client: Annotated[str, Depends(authorize)],  # runs first: no work for unauthorized calls
    assistant: Annotated[RetailAssistant, Depends(assistant_or_503)],
) -> AgentAnswer:
    try:
        return assistant.ask(req.question, req.history)
    except Exception as exc:
        log.exception("chat failed")
        raise upstream_error(exc) from exc


@app.post(
    "/chat/stream",
    operation_id="AskAssistantStream",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"text/event-stream": {}},
            "description": "Server-sent events: status, tool_start, tool_end, token, "
            "blocked, done (full answer with citations) or error",
        },
        401: {"description": "Missing or invalid API key"},
        429: {"description": "Rate limit exceeded"},
    },
)
def chat_stream(
    req: ChatRequest,
    _client: Annotated[str, Depends(authorize)],  # runs first: no work for unauthorized calls
    assistant: Annotated[RetailAssistant, Depends(assistant_or_503)],
) -> StreamingResponse:
    def sse(event: dict) -> str:
        return f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"

    def events():
        try:
            for event in assistant.ask_stream(req.question, req.history):
                if event["type"] == "done":
                    event = {"type": "done", **event["answer"].model_dump()}
                yield sse(event)
        except Exception as exc:  # headers are already sent; report the failure in-band
            log.exception("chat stream failed")
            err = upstream_error(exc)
            yield sse({"type": "error", "status": err.status_code, "detail": err.detail})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
