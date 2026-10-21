"""Tests for decision.py."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# Fakes mirror real interfaces (so some arguments go unused) and note what they were called with.
# pylint: disable=missing-function-docstring,redefined-outer-name,use-implicit-booleaness-not-comparison
import fake_tools
import pytest
from conftest import BASE_TOML, FakeRouter, decision

from decision import check_call, decide, describe
from scenario import load_scenario
from tools import load_tools

RULE = '\n[[rules]]\nwhen = "urgent >= 0.5"\ntell = "Hurry."\n'
TEXT = "I was charged twice"


def urgent(value):
    return {"urgent": {"type": "noul", "noul": value}}


@pytest.fixture(autouse=True)
def no_calls():
    fake_tools.CALLS.clear()


@pytest.fixture
def setup(write_scenario):
    """-> (scenario, tools) for BASE_TOML with one rule, or for the text given."""
    def _setup(text=BASE_TOML + RULE):
        scenario = load_scenario(write_scenario(text))
        return scenario, load_tools(scenario)
    return _setup


def test_decide_returns_route_answers_rules(setup):
    scenario, tools = setup()
    router = FakeRouter(decision("Tech", answers=urgent(0.9)))
    result = decide(TEXT, scenario, router, tools)
    assert (result["route"], result["confidence"], result["uncertain"], result["source"]) == ("Tech", 0.9, False, "clm")
    assert result["probabilities"] == {"Tech": 0.9, "Billing": 0.1}
    assert result["answers"] == urgent(0.9)
    assert result["rules"] == [{"when": "urgent >= 0.5", "tell": "Hurry."}]
    assert (result["informed"], result["actions"]) == ([], [])


def test_facts_are_asked_again_only_for_named_questions(setup):
    text = BASE_TOML.replace("[[actions]]", '[questions.other]\ntype = "noul"\ninstructions = "Other?"\n\n[[actions]]')
    scenario, tools = setup(text)
    router = FakeRouter(decision("Billing", answers={**urgent(0.1), "other": {"type": "noul", "noul": 0.3}}),
                        asks=[urgent(0.8)])
    result = decide(TEXT, scenario, router, tools)
    state, questions = router.ask_calls[0]
    assert list(questions) == ["urgent"]
    assert state == f"{TEXT}\n\nKnown:\n- The queue is A."
    assert result["answers"] == {**urgent(0.8), "other": {"type": "noul", "noul": 0.3}}
    assert result["informed"] == [{"tool": "lookup", "status": "done", "facts": ["The queue is A."]}]


def test_facts_given_to_no_question_ask_nothing(setup):
    scenario, tools = setup(BASE_TOML.replace('questions = ["urgent"]', "questions = []"))
    router = FakeRouter(decision("Billing", answers=urgent(0.9)))
    decide(TEXT, scenario, router, tools)
    assert router.ask_calls == []


def test_facts_do_not_reach_routing(setup):
    scenario, tools = setup()
    router = FakeRouter(decision("Billing", answers=urgent(0.1)), asks=[urgent(0.1)])
    decide(TEXT, scenario, router, tools)
    assert router.states == [TEXT]


def test_action_runs_when_condition_holds(setup):
    scenario, tools = setup()
    router = FakeRouter(decision("Billing", answers=urgent(0.1)), asks=[urgent(0.8)])
    result = decide(TEXT, scenario, router, tools)
    assert result["rules"] == [{"when": "urgent >= 0.5", "tell": "Hurry."}]
    assert result["actions"] == [{"tool": "act", "status": "done", "result": {"id": 1}}]


def test_action_not_run_when_condition_fails(setup):
    scenario, tools = setup()
    router = FakeRouter(decision("Billing", answers=urgent(0.9)), asks=[urgent(0.2)])
    result = decide(TEXT, scenario, router, tools)
    assert (result["rules"], result["actions"]) == ([], [])


def test_uncertain_route_skips_actions(setup):
    scenario, tools = setup()
    router = FakeRouter(decision("Billing", confidence=0.3, answers=urgent(0.1)), asks=[urgent(0.8)])
    result = decide(TEXT, scenario, router, tools)
    assert result["uncertain"] is True
    assert result["informed"][0]["status"] == "done"
    assert result["actions"] == [{"tool": "act", "status": "skipped", "reason": "route uncertain"}]
    assert [name for name, _ in fake_tools.CALLS] == ["lookup"]


def test_failing_tool_is_reported(setup):
    scenario, tools = setup(BASE_TOML.replace('inform = ["lookup"]', 'inform = ["broken"]')
                            .replace("[tools.inform.lookup]", "[tools.inform.broken]")
                            .replace('tool = "act"', 'tool = "broken"').replace('when = "urgent >= 0.5"\n', ""))
    router = FakeRouter(decision("Billing", answers=urgent(0.9)))
    result = decide(TEXT, scenario, router, tools)
    failed = {"status": "failed", "error": "RuntimeError: The service could not be reached."}
    assert result["informed"] == [{"tool": "broken", **failed}]
    assert result["actions"] == [{"tool": "broken", **failed}]
    assert router.ask_calls == []           # no facts, so nothing is asked again
    assert result["route"] == "Billing"


def test_unserialisable_result_is_failed(setup):
    scenario, tools = setup(BASE_TOML.replace('tool = "act"', 'tool = "unserialisable"')
                            .replace('when = "urgent >= 0.5"\n', ""))
    router = FakeRouter(decision("Billing", answers=urgent(0.9)), asks=[urgent(0.9)])
    action = decide(TEXT, scenario, router, tools)["actions"][0]
    assert (action["status"], action["error"].split(":")[0]) == ("failed", "TypeError")


def test_an_informing_tool_must_return_facts(setup, tmp_path):
    (tmp_path / "odd_tools.py").write_text('def lookup(request):\n    return {"facts": "one"}\n\n'
                                           'def act(request):\n    return {}\n\nTOOLS = [lookup, act]\n')
    scenario, tools = setup(BASE_TOML.replace('"fake_tools"', '"odd_tools"'))
    router = FakeRouter(decision("Billing", answers=urgent(0.9)))
    informed = decide(TEXT, scenario, router, tools)["informed"][0]
    assert informed == {"tool": "lookup", "status": "failed",
                        "error": "an informing tool must return {'facts': [sentence, ...]}"}


def test_tool_receives_request_so_far(setup):
    scenario, tools = setup()
    router = FakeRouter(decision("Billing", answers=urgent(0.1)), asks=[urgent(0.8)])
    decide(TEXT, scenario, router, tools)
    informing, acting = (request for _, request in fake_tools.CALLS)
    assert informing == {"text": TEXT, "route": "Billing", "answers": urgent(0.1), "rules": [], "facts": []}
    assert acting == {"text": TEXT, "route": "Billing", "answers": urgent(0.8),
                      "rules": [{"when": "urgent >= 0.5", "tell": "Hurry."}], "facts": ["The queue is A."]}


def test_a_reviewed_route_is_passed_through(setup):
    scenario, tools = setup()
    reviewed = {**decision("Tech"), "source": "reviewed", "similarity": 0.99, "clm_route": "Billing"}
    result = decide(TEXT, scenario, FakeRouter(reviewed), tools)
    assert (result["source"], result["similarity"], result["clm_route"]) == ("reviewed", 0.99, "Billing")


@pytest.mark.parametrize("p, approved", [(0.4, False), (0.5, True)])
def test_check_call_threshold(scenario, p, approved):
    router = FakeRouter(decision("Tech"), tool_p=p)
    call = {"name": "refund", "args": {"invoice": "INV-1"}}
    assert check_call(TEXT, call, scenario, router) == {"p": p, "approved": approved}
    assert router.tool_states == [f'{TEXT}\n\nProposed call: {{"args": {{"invoice": "INV-1"}}, "name": "refund"}}']


def test_describe(setup):
    scenario, _ = setup()
    assert describe(scenario) == {
        "routes": [{"name": "Billing", "description": "Charges and refunds"},
                   {"name": "Tech", "description": "Bugs and outages"}],
        "questions": [{"name": "urgent", "type": "noul", "instructions": "Is this urgent?"}],
        "rules": [{"when": "urgent >= 0.5", "tell": "Hurry."}],
    }


def test_a_reviewed_route_is_not_uncertain(setup):
    scenario, tools = setup()
    reviewed = {**decision("Billing", confidence=0.4, answers=urgent(0.1)), "source": "reviewed",
                "clm_route": "Tech"}
    result = decide(TEXT, scenario, FakeRouter(reviewed, asks=[urgent(0.8)]), tools)
    assert result["uncertain"] is False
    assert result["actions"] == [{"tool": "act", "status": "done", "result": {"id": 1}}]


def test_each_question_hears_only_the_facts_given_to_it(setup, tmp_path):
    (tmp_path / "two_tools.py").write_text(
        'def quiet(request):\n    return {"facts": ["QUIET FACT"]}\n\n'
        'def lookup(request):\n    return {"facts": ["urgent fact"]}\n\n'
        'def act(request):\n    return {}\n\nTOOLS = [quiet, lookup, act]\n')
    text = (BASE_TOML.replace('"fake_tools"', '"two_tools"')
            .replace('inform = ["lookup"]', 'inform = ["quiet", "lookup"]')
            .replace("[tools.inform.lookup]", "[tools.inform.quiet]\nquestions = []\n\n[tools.inform.lookup]"))
    scenario, tools = setup(text)
    router = FakeRouter(decision("Billing", answers=urgent(0.1)), asks=[urgent(0.8)])
    result = decide(TEXT, scenario, router, tools)
    assert router.ask_calls == [(f"{TEXT}\n\nKnown:\n- urgent fact", {"urgent": scenario.questions["urgent"]})]
    assert result["informed"][0]["facts"] == ["QUIET FACT"]


def test_questions_with_different_facts_are_asked_apart(setup, tmp_path):
    (tmp_path / "split_tools.py").write_text(
        'def a(request):\n    return {"facts": ["fact A"]}\n\n'
        'def lookup(request):\n    return {"facts": ["fact B"]}\n\n'
        'def act(request):\n    return {}\n\nTOOLS = [a, lookup, act]\n')
    text = (BASE_TOML.replace('"fake_tools"', '"split_tools"')
            .replace('inform = ["lookup"]', 'inform = ["a", "lookup"]')
            .replace("[tools.inform.lookup]", '[tools.inform.a]\nquestions = ["other"]\n\n[tools.inform.lookup]')
            .replace("[[actions]]", '[questions.other]\ntype = "noul"\ninstructions = "Other?"\n\n[[actions]]'))
    scenario, tools = setup(text)
    other = {"other": {"type": "noul", "noul": 0.6}}
    router = FakeRouter(decision("Billing", answers={**urgent(0.1), **other}), asks=[other, urgent(0.8)])
    decide(TEXT, scenario, router, tools)
    assert sorted((state, list(questions)) for state, questions in router.ask_calls) == [
        (f"{TEXT}\n\nKnown:\n- fact A", ["other"]), (f"{TEXT}\n\nKnown:\n- fact B", ["urgent"])]


def test_tools_receive_context_when_given(setup):
    scenario, tools = setup()
    decide(TEXT, scenario, FakeRouter(decision("Billing", answers=urgent(0.9)), asks=[urgent(0.9)]), tools,
           context={"booking": "PNR1"})
    assert [request["context"] for _, request in fake_tools.CALLS] == [{"booking": "PNR1"}] * 2


def test_tools_get_no_context_key_without_one(setup):
    scenario, tools = setup()
    decide(TEXT, scenario, FakeRouter(decision("Billing", answers=urgent(0.9)), asks=[urgent(0.9)]), tools)
    assert fake_tools.CALLS and all("context" not in request for _, request in fake_tools.CALLS)


def test_context_is_not_in_the_text_clm_reads(setup):
    scenario, tools = setup()
    router = FakeRouter(decision("Tech"))
    decide(TEXT, scenario, router, tools, context={"booking": "PNR1"})
    assert router.states == [TEXT]
