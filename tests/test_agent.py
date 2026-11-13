from conftest import ScriptedChatClient, completion, tool_call

from retail_rag.agent import RetailAssistant


def make_assistant(client, retriever, max_steps=3):
    return RetailAssistant(client, "gpt-test", retriever, "system", "9.9.9", max_steps=max_steps)


def test_agent_calls_tool_then_answers_with_citations(retriever):
    client = ScriptedChatClient(
        [
            completion(
                tool_calls=[
                    tool_call(
                        "c1", "search_products", {"query": "beyaz sneaker", "max_price": 2000}
                    )
                ]
            ),
            completion(content="Ritim Beyaz Deri Sneaker 1899 TL [p-003]. Uydurma [p-999]."),
        ]
    )
    ans = make_assistant(client, retriever).ask("2000 TL altı beyaz sneaker?")

    assert ans.tool_calls[0].name == "search_products"
    assert ans.tool_calls[0].arguments["max_price"] == 2000
    assert "p-003" in ans.tool_calls[0].result_ids
    assert [c.id for c in ans.citations] == ["p-003"]  # hallucinated id dropped
    assert ans.usage.prompt_tokens == 20 and ans.prompt_version == "9.9.9"
    # tool result was fed back to the model
    tool_msgs = [m for m in client.requests[1]["messages"] if m["role"] == "tool"]
    assert tool_msgs and "p-003" in tool_msgs[0]["content"]


def test_agent_forces_answer_after_max_steps(retriever):
    loop = completion(tool_calls=[tool_call("c", "search_policies", {"query": "iade"})])
    client = ScriptedChatClient([loop, loop, completion(content="final")])
    ans = make_assistant(client, retriever, max_steps=2).ask("iade?")
    assert ans.answer == "final"
    assert client.requests[-1]["tool_choice"] == "none"


def test_agent_survives_bad_tool_arguments(retriever):
    bad = completion(tool_calls=[tool_call("c", "search_products", {"no_query": 1})])
    client = ScriptedChatClient([bad, completion(content="Bulamadım.")])
    ans = make_assistant(client, retriever).ask("?")
    assert ans.answer == "Bulamadım."
    assert ans.tool_calls[0].result_ids == []
