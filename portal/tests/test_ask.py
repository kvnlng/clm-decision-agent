"""Tests for Ask: the chat in a page, against the stand-in server and a scripted model, and the testing switch."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
import httpx
import pytest

from clm_portal import chat, web
from conftest import post, sign_in
from test_chat import reply, scripted


def test_ask_shows_every_call_and_result(client, portal):
    models = []

    def make_model(backend, model, url, _http):
        models.append((backend, model, url))
        return scripted([reply(calls=[("decision_vesting", {"years_of_service": 5})]),
                         reply("Fully vested: 100%.")])
    portal.make_model = make_model
    token = sign_in(client)
    page = post(client, "/ask", token, {"question": "Am I vested?", "backend": "openai", "model": "small",
                                        "model_url": "http://model/v1"}).text
    assert "Am I vested?" in page and "[tool: decision_vesting {&quot;years_of_service&quot;: 5}]" in page
    assert "&quot;vested_percent&quot;: 100" in page and "Fully vested: 100%." in page
    assert models == [("openai", "small", "http://model/v1")]
    held = scripted([reply(calls=[("decision_vesting", {"years_of_service": 5})]), reply("Not run.")])
    portal.make_model = lambda *args: held
    post(client, "/ask/clear", token)
    page = post(client, "/ask", token, {"question": "Again?", "backend": "openai", "dry_run": "on"}).text
    assert "dry run, not sent: decision_vesting" in page


def test_a_follow_up_keeps_the_conversation_per_session(client, portal):
    seen: list[dict] = []
    model = scripted([reply("First"), reply("Second"), reply("Third")], seen)
    portal.make_model = lambda *args: model
    token = sign_in(client)
    post(client, "/ask", token, {"question": "One", "backend": "openai"})
    page = post(client, "/ask", token, {"question": "Two", "backend": "openai"}).text
    assert "First" in page and "Second" in page
    assert [item["content"] for item in seen[1]["messages"] if item["role"] == "user"] == ["One", "Two"]
    client.cookies.clear()
    other = sign_in(client, "Someone else")
    page = post(client, "/ask", other, {"question": "Three", "backend": "openai"}).text
    assert [item["content"] for item in seen[2]["messages"] if item["role"] == "user"] == ["Three"]
    assert "One" not in page and "Third" in page


def test_ask_can_hold_back_only_some_tools(client, portal):
    portal.make_model = lambda *args: scripted([
        reply(calls=[("decision_vesting", {"years_of_service": 5}), ("describe", {})]), reply("Done.")])
    token = sign_in(client)
    page = post(client, "/ask", token, {"question": "Vested?", "backend": "openai",
                                        "dry_run_tools": " decision_vesting, "}).text
    assert "dry run, not sent: decision_vesting" in page and "[tool: describe {}]" in page
    assert "Tiny handbook" in page and "Done." in page
    assert 'name="dry_run_tools" value="decision_vesting"' in page


def test_ask_refuses_a_dry_run_tool_the_server_does_not_have(client, portal):
    seen: list[dict] = []
    portal.make_model = lambda *args: scripted([reply("Should not be asked.")], seen)
    token = sign_in(client)
    response = post(client, "/ask", token, {"question": "Vested?", "backend": "openai",
                                            "dry_run_tools": "decision_vestnig"})
    assert response.status_code == 400 and "decision_vestnig, which the server does not have" in response.text
    assert ">Vested?</textarea>" in response.text and not seen and "Should not be asked." not in response.text
    portal.make_model = lambda *args: scripted([reply("Asked.")], seen)
    post(client, "/ask", token, {"question": "Now?", "backend": "openai"})
    assert [item["content"] for item in seen[0]["messages"] if item["role"] == "user"] == ["Now?"]


def test_a_mistyped_tool_does_not_matter_when_every_tool_is_held_back(client, portal):
    portal.make_model = lambda *args: scripted([reply(calls=[("decision_vesting", {"years_of_service": 5})]),
                                                reply("Held.")])
    token = sign_in(client)
    response = post(client, "/ask", token, {"question": "Vested?", "backend": "openai", "dry_run": "on",
                                            "dry_run_tools": "decision_vestnig"})
    assert response.status_code == 200 and "dry run, not sent: decision_vesting" in response.text
    assert "Held." in response.text


def test_ask_shows_results_in_full_only_when_asked(client, portal):
    long_id = "hb-" + "x" * 2000
    calls = [("document_handbook", {"id": long_id})]
    portal.make_model = lambda *args: scripted([reply(calls=calls), reply("Shortened."), reply(calls=calls),
                                                reply("In full.")])
    token = sign_in(client)
    page = post(client, "/ask", token, {"question": "Short?", "backend": "openai"}).text
    assert "more characters" in page and f"The passage (1) of {long_id}, as stored." not in page
    page = post(client, "/ask", token, {"question": "Full?", "backend": "openai", "full": "on"}).text
    last = page[page.index("Full?"):]
    assert "more characters" not in last and f"The passage (1) of {long_id}, as stored." in last
    assert 'name="full" checked' in page


LLM_KEY = "the-model-key-not-a-secret"


@pytest.mark.anyio
async def test_the_model_key_goes_only_to_the_model_url_the_portal_started_with(portal, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", LLM_KEY)
    async with httpx.AsyncClient() as http:
        assert portal.make_model("openai", "", "", http).key == LLM_KEY
        assert portal.make_model("openai", "", chat.OLLAMA_V1 + "/", http).key == LLM_KEY
        assert portal.make_model("openai", "", "http://elsewhere/v1", http).key is None
        portal.options = {"model_url": "http://started/v1"}
        assert portal.make_model("openai", "", "http://started/v1", http).key == LLM_KEY
        assert portal.make_model("openai", "", chat.OLLAMA_V1, http).key is None


def test_ask_says_when_the_model_key_is_not_sent(client, portal, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", LLM_KEY)
    portal.options = {"model_url": "http://started/v1"}
    portal.make_model = lambda *args: scripted([reply("One"), reply("Two")])
    token = sign_in(client)
    assert "sent only to http://started/v1" in client.get("/ask").text
    page = post(client, "/ask", token, {"question": "Here?", "backend": "openai", "model_url": "http://started/v1"})
    assert "was not sent" not in page.text
    page = post(client, "/ask", token, {"question": "There?", "backend": "openai", "model_url": "http://other/v1"})
    assert "LLM_API_KEY was not sent: http://other/v1 is not the model URL the portal started with" in page.text
    assert LLM_KEY not in page.text


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_testing_mode_restarts_the_server_and_every_page_says_so(client, server, environment):
    with environment.open("a") as file:
        file.write("\n[derived]\n")
    token = sign_in(client)
    assert web.TESTING_BANNER not in client.get("/").text
    post(client, "/ask/testing", token, {"on": "1"})
    assert server.starts == [False, True] and server.testing
    for path in ("/", "/ask", "/review", "/documents"):
        assert web.TESTING_BANNER in client.get(path).text, path
    post(client, "/ask/testing", token, {"on": "0"})
    assert server.starts[-1] is False and web.TESTING_BANNER not in client.get("/ask").text


def test_without_a_derived_table_there_is_no_testing_mode(client, server):
    token = sign_in(client)
    assert "/ask/testing" not in client.get("/ask").text
    response = post(client, "/ask/testing", token, {"on": "1"})
    assert response.status_code == 400 and "has no [derived] table" in response.text
    assert server.starts == [False] and not server.testing
