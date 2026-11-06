"""Tests for Ask: the chat in a page, against the stand-in server and a scripted model, and the testing switch."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
import asyncio

import anyio
import httpx
import pytest

from clm_decision.portal import chat, web
from .conftest import KEY, ORIGIN, csrf_of, post, sign_in
from .test_chat import reply, scripted


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


def test_a_model_that_fails_is_shown_as_its_error_not_as_the_server_not_answering(client, portal):
    failing = scripted([])

    async def turn(_messages, _tools):
        raise ValueError("Error code: 401 - invalid x-api-key")
    failing.turn = turn
    portal.make_model = lambda *args: failing
    token = sign_in(client)
    page = post(client, "/ask", token, {"question": "Who?", "backend": "openai"}).text
    assert "[error: " in page and "invalid x-api-key" in page and "not answering" not in page
    seen: list[dict] = []
    failing.turn = scripted([reply("Here.")], seen).turn
    post(client, "/ask", token, {"question": "Now?", "backend": "openai"})
    assert [item["content"] for item in seen[0]["messages"] if item["role"] == "user"] == ["Now?"]


def test_a_model_name_kept_from_one_backend_is_not_sent_to_the_other(client, portal):
    models = []

    def make_model(backend, model, _url, _http):
        models.append((backend, model))
        return scripted([reply("Yes.")])
    portal.make_model = make_model
    portal.options = {"backend": "openai", "model": "granite4:micro"}
    token = sign_in(client)
    post(client, "/ask", token, {"question": "One?", "backend": "openai"})
    assert 'name="model" value="granite4:micro"' in client.get("/ask").text
    for backend, model in (("claude", "granite4:micro"), ("claude", ""), ("openai", ""), ("claude", "claude-x")):
        post(client, "/ask", token, {"question": "Again?", "backend": backend, "model": model})
    assert models == [("openai", "granite4:micro"), ("claude", ""), ("claude", ""), ("openai", "granite4:micro"),
                      ("claude", "claude-x")]


def test_the_started_model_name_shown_on_a_new_conversation_is_not_sent_to_the_other_backend(client, portal):
    models = []

    def make_model(backend, model, _url, _http):
        models.append((backend, model))
        return scripted([reply("Yes.")])
    portal.make_model = make_model
    portal.options = {"backend": "openai", "model": "granite4:micro"}
    token = sign_in(client)
    assert 'name="model" value="granite4:micro"' in client.get("/ask").text
    post(client, "/ask", token, {"question": "First?", "backend": "claude", "model": "granite4:micro"})
    post(client, "/ask", token, {"question": "Second?", "backend": "openai", "model": ""})
    post(client, "/ask/clear", token)
    post(client, "/ask", token, {"question": "Third?", "backend": "claude", "model": "granite4:micro"})
    assert models == [("claude", ""), ("openai", "granite4:micro"), ("claude", "")]


@pytest.mark.anyio
async def test_a_second_question_while_one_is_answered_is_refused_and_the_conversation_stays_whole(portal):
    seen: list[dict] = []
    entered, gate = asyncio.Event(), asyncio.Event()
    model = scripted([reply("First.")], seen)
    answer = model.turn

    async def turn(messages, tools):
        entered.set()
        await gate.wait()
        return await answer(messages, tools)
    model.turn = turn
    portal.make_model = lambda *args: model
    app = web.create_app(portal, KEY, "127.0.0.1", 8400)
    origin = {"Origin": ORIGIN}
    with anyio.fail_after(5):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=ORIGIN) as http:
            await http.post("/sign-in", data={"key": KEY, "name": "Kevin"}, headers=origin)
            form = {"csrf": csrf_of((await http.get("/")).text), "backend": "openai"}
            first = asyncio.create_task(http.post("/ask", data={**form, "question": "One"}, headers=origin))
            await entered.wait()
            second = await http.post("/ask", data={**form, "question": "Two"}, headers=origin)
            cleared = await http.post("/ask/clear", data=form, headers=origin)
            gate.set()
            answered = await first
    assert second.status_code == 409 and "still being answered" in second.text and ">Two</textarea>" in second.text
    assert "reload Ask to see its answer" in second.text
    assert cleared.status_code == 409 and "still being answered" in cleared.text
    assert answered.status_code == 200 and "First." in answered.text
    assert [item["content"] for item in seen[0]["messages"] if item["role"] == "user"] == ["One"]
    assert [question for question, _ in next(iter(portal.sessions.values())).ask["turns"]] == ["One"]


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


def test_testing_mode_turns_on_without_a_reload_and_every_page_says_so(client, server, environment):
    with environment.open("a") as file:
        file.write("\n[derived]\n")
    token = sign_in(client)
    assert web.TESTING_BANNER not in client.get("/").text
    post(client, "/ask/testing", token, {"on": "1"})
    assert server.testing and not server.reloads
    for path in ("/", "/ask", "/review", "/documents"):
        assert web.TESTING_BANNER in client.get(path).text, path
    post(client, "/ask/testing", token, {"on": "0"})
    assert not server.testing and web.TESTING_BANNER not in client.get("/ask").text


def test_without_a_derived_table_there_is_no_testing_mode(client, server):
    token = sign_in(client)
    assert "/ask/testing" not in client.get("/ask").text
    response = post(client, "/ask/testing", token, {"on": "1"})
    assert response.status_code == 400 and "has no [derived] table" in response.text
    assert not server.testing
