import pytest
from conftest import ScriptedChatClient, completion, tool_call
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from retail_rag.agent import RetailAssistant

_exporter = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(_exporter))
trace.set_tracer_provider(_provider)


@pytest.fixture
def spans():
    _exporter.clear()
    yield lambda: {s.name: s for s in _exporter.get_finished_spans()}


def test_agent_emits_genai_spans(retriever, spans):
    client = ScriptedChatClient(
        [
            completion(tool_calls=[tool_call("c1", "search_policies", {"query": "iade"})]),
            completion(content="30 gün [faq-iade-suresi-ve-kosullari-0]."),
        ]
    )
    question = "İade süresi nedir? özel bilgi 12345"
    RetailAssistant(client, "gpt-test", retriever, "system", "7.7.7").ask(question)
    got = spans()

    agent = got["invoke_agent retail-assistant"]
    assert agent.attributes["retail_rag.prompt_version"] == "7.7.7"
    assert agent.attributes["gen_ai.usage.input_tokens"] == 20
    assert agent.attributes["retail_rag.citations"] == 1

    tool = got["execute_tool search_policies"]
    assert tool.attributes["retail_rag.tool.result_count"] > 0
    assert tool.parent.span_id == agent.context.span_id

    retrieve = got["retrieve"]
    assert retrieve.attributes["retail_rag.filter"] == "source eq 'faq'"
    assert retrieve.parent.span_id == tool.context.span_id
    assert got["chat gpt-test"].attributes["gen_ai.usage.output_tokens"] == 5

    # the customer's text is never recorded
    for s in _exporter.get_finished_spans():
        assert all("12345" not in str(v) for v in s.attributes.values())
