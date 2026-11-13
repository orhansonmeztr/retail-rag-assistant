# Using the assistant from Microsoft Copilot Studio

The API is exposed to Copilot Studio as a **REST API tool** (a Power Platform custom
connector under the hood). The copilot keeps the conversation; our API does the retrieval,
grounding and citation work.

## Prerequisites

- The API running on Azure Container Apps (see [deploy.md](deploy.md)) with `API_KEYS` set.
- A Copilot Studio environment (a trial works).

## Steps

1. Put your Container Apps host name into `copilot/openapi-v2.yaml` (`host:`).
2. In Copilot Studio open your agent → **Tools** → **Add a tool** → **New tool** →
   **REST API**, and upload `copilot/openapi-v2.yaml`.
3. Authentication: **API key**, parameter label `API key`, parameter name `X-API-Key`,
   location **Header**. Create the connection with one of the keys from `API_KEYS`.
4. Select the `AskAssistant` operation. Keep its description; the copilot's
   orchestrator uses it to decide when to call the tool.
5. In the agent's instructions add for example:
   *"For any question about products, orders, returns, shipping, payment, loyalty points or
   gift cards, call AskAssistant with the user's question and reply with its `answer`.
   Do not answer these topics from your own knowledge."*
6. Test in the test pane: "Kozmetik ürününü açtım, iade edebilir miyim?" should produce
   a tool call and an answer citing `faq-iade-suresi-ve-kosullari-0`.

## Notes

- Swagger 2.0 is used because Power Platform custom connectors are defined in that format.
  FastAPI's own `/openapi.json` (OpenAPI 3) is still available for other clients.
- The `blocked` flag lets a topic branch to a hand-off message when the content filter
  refused a request.
- Rate limits (HTTP 429) are per API key; give the copilot connection its own key.
