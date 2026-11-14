import sys
import types

from retail_rag import telemetry


def test_connection_string_from_settings_reaches_exporter(monkeypatch):
    """A value that only lives in .env is not in os.environ; it must be passed explicitly."""
    seen = {}
    fake = types.ModuleType("azure.monitor.opentelemetry")
    fake.configure_azure_monitor = lambda **kw: seen.update(kw)
    monkeypatch.setitem(sys.modules, "azure.monitor.opentelemetry", fake)
    monkeypatch.delenv("APPLICATIONINSIGHTS_CONNECTION_STRING", raising=False)
    monkeypatch.setattr(telemetry, "_configured", False)

    assert telemetry.configure_telemetry("InstrumentationKey=test") is True
    assert seen["connection_string"] == "InstrumentationKey=test"


def test_no_connection_string_is_a_no_op(monkeypatch):
    monkeypatch.delenv("APPLICATIONINSIGHTS_CONNECTION_STRING", raising=False)
    monkeypatch.setattr(telemetry, "_configured", False)
    assert telemetry.configure_telemetry(None) is False
