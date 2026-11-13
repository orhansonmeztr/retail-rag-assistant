"""Request-level protection: API keys, rate limiting and Azure AI Content Safety Prompt Shields."""

from __future__ import annotations

import logging
import secrets
import threading
import time
from collections import defaultdict, deque

import httpx

log = logging.getLogger(__name__)


def parse_api_keys(raw: str | None) -> list[str]:
    return [k.strip() for k in (raw or "").split(",") if k.strip()]


def key_is_valid(candidate: str | None, keys: list[str]) -> bool:
    if not candidate:
        return False
    # compare against every key in constant time so timing does not reveal a prefix match
    return any(secrets.compare_digest(candidate.encode(), k.encode()) for k in keys)


class RateLimiter:
    """Sliding one-minute window per client. In-process: with several replicas each keeps
    its own counts, so put a gateway (e.g. API Management) in front for a global limit."""

    def __init__(self, per_minute: int, clock=time.monotonic):
        self.per_minute = per_minute
        self.clock = clock
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, client: str) -> int | None:
        """None if allowed, otherwise seconds until the client may retry."""
        if self.per_minute <= 0:
            return None
        now = self.clock()
        with self._lock:
            hits = self._hits[client]
            while hits and now - hits[0] >= 60:
                hits.popleft()
            if len(hits) >= self.per_minute:
                return max(1, int(60 - (now - hits[0])) + 1)
            hits.append(now)
            return None


class PromptShield:
    """Azure AI Content Safety Prompt Shields: detects jailbreak / prompt-injection attempts
    in the user's message before it reaches the model."""

    API_VERSION = "2024-09-01"

    def __init__(self, endpoint: str, api_key: str, timeout_s: float = 5.0, client=None):
        self.url = f"{endpoint.rstrip('/')}/contentsafety/text:shieldPrompt"
        self.headers = {"Ocp-Apim-Subscription-Key": api_key}
        self.client = client or httpx.Client(timeout=timeout_s)

    def attack_detected(self, user_prompt: str, documents: list[str] | None = None) -> bool:
        resp = self.client.post(
            self.url,
            params={"api-version": self.API_VERSION},
            headers=self.headers,
            json={"userPrompt": user_prompt, "documents": documents or []},
        )
        resp.raise_for_status()
        body = resp.json()
        user = body.get("userPromptAnalysis", {}).get("attackDetected", False)
        docs = any(d.get("attackDetected") for d in body.get("documentsAnalysis", []))
        return bool(user or docs)
