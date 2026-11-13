import httpx
import openai
import pytest
from azure.core.exceptions import ServiceResponseTimeoutError
from conftest import ScriptedChatClient
from fastapi.testclient import TestClient

from retail_rag.agent import RetailAssistant
from retail_rag.api import app, assistant_or_503, get_api_keys
from retail_rag.search_backend import AzureSearchBackend

_req = httpx.Request("POST", "https://x")


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (openai.APITimeoutError(request=_req), 504),
        (ServiceResponseTimeoutError("read timeout"), 504),
        (openai.RateLimitError("busy", response=httpx.Response(429, request=_req), body=None), 503),
        (RuntimeError("boom"), 502),
    ],
)
def test_upstream_errors_map_to_retry_aware_status(retriever, error, status):
    class Failing(ScriptedChatClient):
        def _create(self, **kwargs):
            raise error

    app.dependency_overrides[assistant_or_503] = lambda: RetailAssistant(
        Failing([]), "m", retriever, "s", "1"
    )
    app.dependency_overrides[get_api_keys] = lambda: ()
    try:
        r = TestClient(app).post("/chat", json={"question": "x"})
        assert r.status_code == status
        if status == 503:
            assert r.headers["Retry-After"] == "10"
        # the stream reports the same status in-band
        s = TestClient(app).post("/chat/stream", json={"question": "x"})
        assert f'"status": {status}' in s.text
    finally:
        app.dependency_overrides.clear()


def _pipeline(client):
    """Find the azure-core pipeline; its attribute path differs between SDK versions."""
    obj = client
    for _ in range(4):
        if hasattr(obj, "_pipeline"):
            return obj._pipeline
        obj = obj._client
    raise AssertionError("pipeline not found")


def test_search_client_has_short_timeouts_and_retries():
    from azure.core.credentials import AzureKeyCredential
    from azure.core.pipeline.policies import RetryPolicy

    b = AzureSearchBackend(
        "https://x.search.windows.net", "idx", AzureKeyCredential("k"), timeout_s=7, max_retries=2
    )
    pipeline = _pipeline(b.client)
    [retry] = [p for p in pipeline._impl_policies if isinstance(p, RetryPolicy)]
    assert retry.total_retries == 2
    conn = pipeline._transport.connection_config
    assert conn.read_timeout == 7 and conn.timeout == 5
