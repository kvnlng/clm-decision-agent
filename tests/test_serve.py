"""Tests for serve.py."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# Fakes mirror real interfaces (so some arguments go unused).
# pylint: disable=missing-function-docstring,missing-class-docstring,redefined-outer-name
# pylint: disable=unused-argument,use-implicit-booleaness-not-comparison
import threading

import httpx
import pytest
from conftest import CONFIRM_TOML, INTAKE_TOML, FakeGenerator, FakeRouter, decision, tool_call

import chat
import serve
from generator import GeneratorError
from graph import build_graph
from scenario import load_scenario
from serve import Conversation, make_server


@pytest.fixture
def serving(write_scenario):
    """Start a server for a scenario and a scripted generator; -> (base URL, the conversation)."""
    servers = []

    def _serve(gen, toml=CONFIRM_TOML, router=None):
        scenario = load_scenario(write_scenario(toml))
        graph = build_graph(scenario, router or FakeRouter(decision("Billing"), tool_p=0.82), gen)
        conversation = Conversation(graph, scenario)
        server = make_server(conversation, port=0)
        threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()
        servers.append(server)
        return f"http://127.0.0.1:{server.server_address[1]}", conversation
    yield _serve
    for server in servers:
        server.shutdown()
        server.server_close()


REFUND = tool_call("issue_refund", invoice_number="INV-1", amount=5.0)


def test_a_message_gets_a_reply_with_the_decision(serving):
    url, _ = serving(FakeGenerator("Your invoice is fine."))
    answer = httpx.post(f"{url}/message", json={"text": "is INV-1 right?"})

    assert answer.status_code == 200
    body = answer.json()
    assert body["status"] == "reply" and body["reply"] == "Your invoice is fine."
    assert body["decision"]["route"] == "Billing" and body["decision"]["confidence"] == 0.9
    assert "state_text" not in body["decision"]
    assert body["tool_checks"] == [] and body["intake"] is None


def test_a_confirm_tool_waits_and_a_yes_runs_it(serving):
    url, _ = serving(FakeGenerator(REFUND, "Refunded."))
    waiting = httpx.post(f"{url}/message", json={"text": "refund INV-1"}).json()

    assert waiting["status"] == "waiting"
    assert waiting["check"]["calls"] == [{"name": "issue_refund", "args": {"invoice_number": "INV-1", "amount": 5.0}}]
    assert waiting["check"]["p"] == 0.82 and "state_text" not in waiting["check"]
    assert httpx.get(f"{url}/").json()["waiting"] is True

    done = httpx.post(f"{url}/confirm", json={"approved": True}).json()
    assert done["status"] == "reply" and done["reply"] == "Refunded."
    assert done["tool_checks"][0]["source"] == "user" and done["tool_checks"][0]["approved"] is True
    assert httpx.get(f"{url}/").json()["waiting"] is False


def test_a_no_declines_the_call_and_the_turn_still_replies(serving):
    gen = FakeGenerator(REFUND, "No refund was made. What would you like to change?")
    url, _ = serving(gen)
    httpx.post(f"{url}/message", json={"text": "refund INV-1"})
    done = httpx.post(f"{url}/confirm", json={"approved": False}).json()

    assert done["status"] == "reply" and done["tool_checks"][0]["approved"] is False
    assert "chose not to run" in gen.calls[-1]["messages"][-1].content


def test_a_message_while_a_call_waits_is_refused_and_the_call_still_waits(serving):
    url, _ = serving(FakeGenerator(REFUND, "Refunded."))
    httpx.post(f"{url}/message", json={"text": "refund INV-1"})
    refused = httpx.post(f"{url}/message", json={"text": "and another thing"})

    assert refused.status_code == 409 and "/confirm" in refused.json()["error"]
    assert httpx.post(f"{url}/confirm", json={"approved": True}).json()["reply"] == "Refunded."


def test_confirm_with_nothing_waiting_is_refused(serving):
    url, _ = serving(FakeGenerator())
    assert httpx.post(f"{url}/confirm", json={"approved": True}).status_code == 409


@pytest.mark.parametrize("path, content", [
    ("/message", b"not json"), ("/message", b'["a list"]'), ("/message", b'{"text": "  "}'),
    ("/message", b"{}"), ("/confirm", b'{"approved": "yes"}'), ("/message", b""),
])
def test_a_bad_body_is_a_400_and_starts_no_turn(serving, path, content):
    gen = FakeGenerator()
    url, _ = serving(gen)
    answer = httpx.post(f"{url}{path}", content=content)

    assert answer.status_code == 400 and answer.json()["error"]
    assert gen.calls == []


def test_unknown_paths_and_methods_are_404(serving):
    url, _ = serving(FakeGenerator())
    assert httpx.post(f"{url}/nope", json={}).status_code == 404
    assert httpx.get(f"{url}/message").status_code == 404


def test_the_conversation_is_carried_between_requests_and_new_discards_the_draft(serving):
    gen = FakeGenerator('{"invoice": "INV-1"}', "What is wrong with it?",
                        '{"problem": "charged twice"}', "What is the invoice?")
    url, conversation = serving(gen, toml=INTAKE_TOML)
    first = httpx.post(f"{url}/message", json={"text": "about INV-1"}).json()

    assert first["intake"] == {"filled": 1, "total": 2, "missing": ["problem"]}
    status = httpx.get(f"{url}/").json()
    assert status["draft_route"] == "Billing" and status["draft"] == {"invoice": "INV-1"}
    assert set(status["routes"]) == {"Billing", "Tech"}

    assert httpx.post(f"{url}/new").json() == {"status": "new"}
    assert httpx.get(f"{url}/").json()["draft"] == {}
    assert len(conversation.session["messages"]) == 2      # the conversation itself is kept

    second = httpx.post(f"{url}/message", json={"text": "charged twice"}).json()
    assert second["intake"]["missing"] == ["invoice"]      # the discarded invoice is not filled back in


def test_new_abandons_a_waiting_call(serving):
    url, _ = serving(FakeGenerator(REFUND, "Hello again."))
    httpx.post(f"{url}/message", json={"text": "refund INV-1"})
    httpx.post(f"{url}/new")

    assert httpx.post(f"{url}/confirm", json={"approved": True}).status_code == 409
    assert httpx.post(f"{url}/message", json={"text": "hello"}).json()["reply"] == "Hello again."


def test_a_model_failure_is_a_502_and_the_conversation_is_unchanged(serving):
    class Failing:
        def generate(self, system, messages, tools=None):
            raise GeneratorError("the model server is not running")

    url, conversation = serving(Failing())
    answer = httpx.post(f"{url}/message", json={"text": "hello"})

    assert answer.status_code == 502 and answer.json() == {"error": "the model server is not running"}
    assert conversation.session["messages"] == [] and conversation.waiting is None


def test_main_reports_a_bad_scenario_and_does_not_accept_review(tmp_path, capsys):
    assert serve.main([str(tmp_path / "nope.toml")]) == 2
    assert capsys.readouterr().err.startswith("error: ")
    with pytest.raises(SystemExit):
        serve.main([str(tmp_path / "nope.toml"), "--review"])


def test_main_loads_in_run_mode_and_says_where_it_listens(write_scenario, monkeypatch, capsys):
    class Router:
        def load_verdicts(self, records, scenario):
            return 0

    served = []
    monkeypatch.setattr(chat, "make_router", lambda scenario: (Router(), "stub"))
    monkeypatch.setattr(chat, "make_generator", lambda scenario: FakeGenerator())
    monkeypatch.setattr(serve.HTTPServer, "serve_forever", lambda self: served.append(self.server_address))

    assert serve.main([str(write_scenario()), "--port", "0"]) == 0
    out = capsys.readouterr().out
    assert served and served[0][0] == "127.0.0.1"
    assert "Run mode" in out and "this computer only" in out and "no authentication" in out
