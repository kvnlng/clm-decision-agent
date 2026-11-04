"""Tests for Ask: the chat in a page, against the stand-in server and a scripted model, and the testing switch."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
from clm_portal import web
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


def test_testing_mode_restarts_the_server_and_every_page_says_so(client, server):
    token = sign_in(client)
    assert web.TESTING_BANNER not in client.get("/").text
    post(client, "/ask/testing", token, {"on": "1"})
    assert server.starts == [False, True] and server.testing
    for path in ("/", "/ask", "/review", "/documents"):
        assert web.TESTING_BANNER in client.get(path).text, path
    post(client, "/ask/testing", token, {"on": "0"})
    assert server.starts[-1] is False and web.TESTING_BANNER not in client.get("/ask").text
