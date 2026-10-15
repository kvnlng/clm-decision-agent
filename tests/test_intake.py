"""Tests for the intake: collecting a route's fields over several turns, then acting."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# Fakes mirror real interfaces (so some arguments go unused) and note what they were called with.
# pylint: disable=missing-function-docstring,missing-class-docstring,redefined-outer-name
# pylint: disable=unused-argument,attribute-defined-outside-init,use-implicit-booleaness-not-comparison
from uuid import uuid4

import pytest
from conftest import (CATALOGUE_ENTRIES, CATALOGUE_TOML, INTAKE_TOML, FakeGenerator, FakeRouter, decision,
                      tool_call)
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.types import Command

from generator import GeneratorError
from graph import USER_DECLINED, build_graph, parse_fields
from scenario import load_scenario

KEYS = ("messages", "draft", "draft_route", "draft_start", "follow_up")
FIELDS = {"invoice": "Which invoice", "problem": "What is wrong with it"}


@pytest.fixture
def intake(write_scenario):
    return load_scenario(write_scenario(INTAKE_TOML))


def new_session():
    return {"messages": [], "draft": {}, "draft_route": None, "draft_start": 0, "follow_up": False}


def turn(graph, session, text, approve=True, answers=None):
    """One turn. Returns (result, session for the next turn, kinds of pause met)."""
    config = {"configurable": {"thread_id": uuid4().hex}}
    out = graph.invoke({**session, "messages": session["messages"] + [HumanMessage(content=text)]}, config)
    kinds = []
    while out.get("__interrupt__"):
        kind = out["__interrupt__"][0].value["kind"]
        kinds.append(kind)
        resume = (answers or {}).get(kind, {"approved": approve})
        out = graph.invoke(Command(resume=resume), config)
    return out, {key: out.get(key) for key in KEYS}, kinds


def submit_call(name="submit_form"):
    return tool_call(name, summary="INV-1 charged twice")


@pytest.mark.parametrize("text, expected", [
    ('{"invoice": "INV-1", "problem": null}', {"invoice": "INV-1"}),
    ('Here you go:\n```json\n{"invoice": " INV-1 ", "problem": "charged twice"}\n```',
     {"invoice": "INV-1", "problem": "charged twice"}),
    ('{"invoice": 2041, "problem": "Not sure", "other": "x"}', {"invoice": "2041"}),
    ('{"invoice": "unknown", "problem": ""}', {}),
    ('{"invoice": "INV-1", "problem": "not sure yet, my manager asked me to get it"}', {"invoice": "INV-1"}),
    ('{"invoice": "I don\'t know the number", "problem": "No idea"}', {}),
    ('{"invoice": ["a"], "problem": true}', {}),
])
def test_parse_fields_keeps_stated_values_only(text, expected):
    assert parse_fields(text, FIELDS) == expected


@pytest.mark.parametrize("text", ["I could not find anything.", '{"invoice": "INV-1"', "[1, 2]"])
def test_parse_fields_returns_none_without_a_json_object(text):
    assert parse_fields(text, FIELDS) is None


def test_first_turn_extracts_what_was_said_and_asks_for_the_rest(intake):
    gen = FakeGenerator('{"invoice": "INV-1", "problem": null}', "What is wrong with it?")
    out, session, _ = turn(build_graph(intake, FakeRouter(decision("Billing")), gen), new_session(), "about INV-1")

    assert out["draft"] == {"invoice": "INV-1"}
    assert out["draft_route"] == "Billing"
    assert out["intake"] == {"filled": 1, "total": 2, "missing": ["problem"]}
    assert out["messages"][-1].content == "What is wrong with it?"
    extract, ask_for = gen.calls[0], gen.calls[1]
    assert len(gen.calls) == 2
    assert '- "invoice": Which invoice' in extract["system"] and extract["tools"] is None
    assert "Already have:\n- Which invoice: INV-1" in ask_for["system"]
    assert "Still needed:\n- What is wrong with it" in ask_for["system"]
    assert ask_for["system"].startswith("You handle billing.")
    assert ask_for["tools"] is None
    assert session["draft"] == {"invoice": "INV-1"}


def test_an_open_intake_keeps_its_route_and_completes(intake):
    first = "User: about INV-1"
    second = "User: about INV-1\nAssistant: What is wrong with it?\nUser: charged twice"
    router = FakeRouter({first: decision("Billing"), second: decision("Tech")})
    gen = FakeGenerator('{"invoice": "INV-1"}', "What is wrong with it?",
                        '{"problem": "charged twice"}', submit_call(), "Submitted as F-1.")
    graph = build_graph(intake, router, gen)
    _, session, _ = turn(graph, new_session(), "about INV-1")
    out, session, kinds = turn(graph, session, "charged twice")

    assert (out["decision"]["route"], out["decision"]["source"], out["decision"]["clm_route"]) == \
        ("Billing", "intake", "Tech")
    assert kinds == ["confirm"]
    act, done = gen.calls[3], gen.calls[4]
    assert "All the details needed are collected:\n- Which invoice: INV-1\n- What is wrong with it: charged twice" \
        in act["system"]
    assert [t.name for t in act["tools"]] == ["lookup_invoice", "submit_form", "broken_submit"]
    assert "has been submitted" in done["system"] and done["tools"] is None
    assert any(isinstance(m, ToolMessage) and m.name == "submit_form" and "F-1" in m.content for m in out["messages"])
    assert out["messages"][-1].content == "Submitted as F-1."
    assert (out["draft"], out["draft_route"], out["submitted"]) == ({}, None, True)
    assert (session["draft"], session["draft_route"]) == ({}, None)


def test_extraction_that_is_not_json_leaves_the_draft_unchanged(intake):
    gen = FakeGenerator('{"invoice": "INV-1"}', "What is wrong with it?",
                        "Sorry, I am not sure what you mean.", "Could you say what is wrong with it?")
    graph = build_graph(intake, FakeRouter(decision("Billing")), gen)
    _, session, _ = turn(graph, new_session(), "about INV-1")
    out, _, _ = turn(graph, session, "hmm")

    assert out["draft"] == {"invoice": "INV-1"}
    assert out["intake"]["missing"] == ["problem"]
    assert out["messages"][-1].content == "Could you say what is wrong with it?"


def test_a_later_answer_replaces_an_earlier_value(intake):
    gen = FakeGenerator('{"invoice": "INV-1"}', "What is wrong?", '{"invoice": "INV-2"}', "And what is wrong?")
    graph = build_graph(intake, FakeRouter(decision("Billing")), gen)
    _, session, _ = turn(graph, new_session(), "about INV-1")
    out, _, _ = turn(graph, session, "sorry, I meant INV-2")
    assert out["draft"] == {"invoice": "INV-2"}


def test_refusing_confirmation_keeps_the_draft(intake):
    gen = FakeGenerator('{"invoice": "INV-1", "problem": "charged twice"}', submit_call(), "What should I change?")
    out, session, kinds = turn(build_graph(intake, FakeRouter(decision("Billing")), gen), new_session(),
                               "INV-1 was charged twice", approve=False)

    assert kinds == ["confirm"]
    assert any(isinstance(m, ToolMessage) and m.content == USER_DECLINED for m in out["messages"])
    assert not any(isinstance(m, ToolMessage) and "F-1" in m.content for m in out["messages"])
    assert session["draft"] == {"invoice": "INV-1", "problem": "charged twice"}
    assert session["draft_route"] == "Billing"
    assert not out["submitted"]


def test_a_failed_submission_keeps_the_draft_and_still_replies(intake):
    gen = FakeGenerator('{"invoice": "INV-1", "problem": "charged twice"}', submit_call("broken_submit"),
                        "The service is down; your details are saved.")
    out, session, _ = turn(build_graph(intake, FakeRouter(decision("Billing")), gen), new_session(),
                           "INV-1 was charged twice")

    failed = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert failed[0].status == "error" and "could not be reached" in failed[0].content
    assert session["draft"] == {"invoice": "INV-1", "problem": "charged twice"}
    assert not out["submitted"]
    assert out["messages"][-1].content == "The service is down; your details are saved."


def test_a_new_request_after_submission_is_not_filled_from_the_old_conversation(intake):
    gen = FakeGenerator('{"invoice": "INV-1", "problem": "charged twice"}', submit_call(), "Submitted as F-1.",
                        '{"invoice": "INV-9"}', "What is wrong with INV-9?")
    graph = build_graph(intake, FakeRouter(decision("Billing")), gen)
    _, session, _ = turn(graph, new_session(), "INV-1 was charged twice")
    out, session, _ = turn(graph, session, "now INV-9 please")

    assert [m.content for m in gen.calls[3]["messages"]] == ["now INV-9 please"]
    assert out["draft"] == {"invoice": "INV-9"}
    assert session["draft_start"] == len(out["messages"]) - 2


def test_review_mode_asks_for_the_route_only_when_an_intake_starts(intake):
    records = []
    gen = FakeGenerator('{"invoice": "INV-1"}', "What is wrong with it?",
                        '{"problem": "charged twice"}', "I have everything.")
    graph = build_graph(intake, FakeRouter(decision("Billing")), gen, review=True, record=records.append)
    answers = {"route": {"route": "Billing"}, "reply": {"acceptable": True, "note": ""}}
    _, session, first = turn(graph, new_session(), "about INV-1", answers=answers)
    _, _, second = turn(graph, session, "charged twice", answers=answers)

    assert first == ["route", "reply"]
    assert second == ["reply"]
    assert [r["kind"] for r in records] == ["route", "reply", "reply"]


def test_a_route_without_fields_is_unaffected(intake):
    gen = FakeGenerator("Try restarting.")
    out, session, _ = turn(build_graph(intake, FakeRouter(decision("Tech")), gen), new_session(), "it crashes")
    assert len(gen.calls) == 1
    assert (session["draft"], session["draft_route"], out["intake"]) == ({}, None, None)


def test_after_a_refusal_the_generator_is_told_nothing_was_submitted_and_gets_no_tools(intake):
    gen = FakeGenerator('{"invoice": "INV-1", "problem": "charged twice"}', submit_call(), "What should I change?")
    turn(build_graph(intake, FakeRouter(decision("Billing")), gen), new_session(), "INV-1 was charged twice",
         approve=False)
    after = gen.calls[2]
    assert "was not submitted" in after["system"] and "submit the request with" not in after["system"]
    assert after["tools"] is None


def test_after_a_failed_submission_the_generator_is_not_told_to_submit_again(intake):
    gen = FakeGenerator('{"invoice": "INV-1", "problem": "charged twice"}', submit_call("broken_submit"), "Down.")
    turn(build_graph(intake, FakeRouter(decision("Billing")), gen), new_session(), "INV-1 was charged twice")
    after = gen.calls[2]
    assert "was not submitted" in after["system"] and after["tools"] is None


def test_a_lookup_before_submitting_still_leads_to_the_submitting_tool(intake):
    gen = FakeGenerator('{"invoice": "INV-1", "problem": "charged twice"}',
                        tool_call("lookup_invoice", invoice_number="INV-1"), submit_call(), "Submitted as F-1.")
    out, _, kinds = turn(build_graph(intake, FakeRouter(decision("Billing")), gen), new_session(),
                         "INV-1 was charged twice")
    assert kinds == ["confirm"] and out["submitted"]
    assert "submit the request with" in gen.calls[2]["system"] and gen.calls[2]["tools"]


def test_a_generator_failure_after_submission_still_reports_it_and_closes_the_draft(intake):
    class FailsAtTheEnd(FakeGenerator):
        def generate(self, system, messages, tools=None):
            if "has been submitted" in system:
                raise GeneratorError("Cannot reach Ollama.")
            return super().generate(system, messages, tools)

    gen = FailsAtTheEnd('{"invoice": "INV-1", "problem": "charged twice"}', submit_call())
    out, session, _ = turn(build_graph(intake, FakeRouter(decision("Billing")), gen), new_session(),
                           "INV-1 was charged twice")
    assert "F-1" in out["messages"][-1].content and "submitted" in out["messages"][-1].content
    assert (session["draft"], session["draft_route"]) == ({}, None)


def test_details_given_before_a_clarifying_question_are_not_lost(intake):
    first = "User: INV-1 was charged twice"
    second = first + "\nAssistant: Billing or tech?\nUser: billing"
    router = FakeRouter({first: decision("Billing", 0.3), second: decision("Billing", 0.9)})
    gen = FakeGenerator("Billing or tech?", '{"invoice": "INV-1", "problem": "charged twice"}', "I have everything.")
    graph = build_graph(intake, router, gen)
    _, session, _ = turn(graph, new_session(), "INV-1 was charged twice")
    out, _, _ = turn(graph, session, "billing")
    assert [m.content for m in gen.calls[1]["messages"]] == ["INV-1 was charged twice", "Billing or tech?", "billing"]
    assert out["draft"] == {"invoice": "INV-1", "problem": "charged twice"}


def test_thanks_after_a_submission_gets_a_reply_and_does_not_open_a_new_request(intake):
    gen = FakeGenerator('{"invoice": "INV-1", "problem": "charged twice"}', submit_call(), "Submitted as F-1.",
                        '{"invoice": null, "problem": null}', "You're welcome.",
                        '{"invoice": "INV-9"}', "What is wrong with INV-9?")
    graph = build_graph(intake, FakeRouter(decision("Billing")), gen)
    _, session, _ = turn(graph, new_session(), "INV-1 was charged twice")
    assert session["follow_up"] is True

    out, session, _ = turn(graph, session, "thanks!")
    assert out["messages"][-1].content == "You're welcome."
    assert (out["draft"], out["draft_route"], out["intake"]) == ({}, None, None)
    assert "has already been submitted" in gen.calls[4]["system"] and gen.calls[4]["tools"] is None

    out, session, _ = turn(graph, session, "now INV-9 please")
    assert (out["draft"], out["draft_route"]) == ({"invoice": "INV-9"}, "Billing")
    assert session["follow_up"] is False


def test_a_vague_first_message_still_opens_a_request(intake):
    gen = FakeGenerator("{}", "Which invoice is it?")
    out, _, _ = turn(graph := build_graph(intake, FakeRouter(decision("Billing")), gen), new_session(), "I need help")
    assert graph and out["draft_route"] == "Billing" and out["intake"]["filled"] == 0


def test_a_second_request_is_worked_on_without_the_finished_one_in_view(intake):
    gen = FakeGenerator('{"invoice": "INV-1", "problem": "charged twice"}', submit_call(), "Submitted as F-1.",
                        '{"invoice": "INV-9"}', "What is wrong with INV-9?",
                        '{"problem": "wrong amount"}', submit_call(), "Submitted as F-1.")
    graph = build_graph(intake, FakeRouter(decision("Billing")), gen)
    _, session, _ = turn(graph, new_session(), "INV-1 was charged twice")
    _, session, _ = turn(graph, session, "now INV-9 please")
    asked = gen.calls[4]
    assert [m.content for m in asked["messages"]] == ["now INV-9 please"]

    turn(graph, session, "wrong amount")
    acted = gen.calls[6]
    assert [m.content for m in acted["messages"]] == ["now INV-9 please", "What is wrong with INV-9?", "wrong amount"]


# The catalogue: CLM picks an invoice on the Billing intake, and a confident pick's facts inform the rest

URGENT = {"urgent": {"type": "noul", "noul": 0.8}}
NOTHING = '{"invoice": null, "problem": null}'


@pytest.fixture
def catalogue_intake(write_scenario):
    return load_scenario(write_scenario(INTAKE_TOML + CATALOGUE_TOML))


def pick(choice, confidence):
    others = {entry["name"]: 1 - confidence for entry in CATALOGUE_ENTRIES if entry["name"] != choice}
    return {"catalogue": {"type": "choice", "choice": choice, "confidence": confidence,
                          "probabilities": {choice: confidence, **others}}}


def with_catalogue(scenario, router, gen):
    return build_graph(scenario, router, gen, catalogue=CATALOGUE_ENTRIES)


def test_a_confident_pick_fills_the_field_and_its_facts_inform_questions_and_prompt(catalogue_intake):
    router = FakeRouter(decision("Billing"), asks=[pick("INV-1", 0.9), URGENT])
    gen = FakeGenerator('{"invoice": "the march one", "problem": null}', "What is wrong with it?")
    out, _, _ = turn(with_catalogue(catalogue_intake, router, gen), new_session(), "my march invoice")

    assert out["draft"] == {"invoice": "INV-1"}                     # CLM's pick overrides the extraction
    assert out["decision"]["catalogue"] == {"choice": "INV-1", "confidence": 0.9, "picked": True,
                                            "probabilities": pick("INV-1", 0.9)["catalogue"]["probabilities"]}
    assert out["decision"]["answers"] == URGENT
    assert len(router.ask_calls) == 2
    (pick_state, pick_questions), (state, questions) = router.ask_calls[0], router.ask_calls[1]
    assert pick_state == "User: my march invoice"
    assert pick_questions["catalogue"]["criteria"] == {"INV-1": "March invoice", "INV-2": "April invoice"}
    assert pick_questions["catalogue"]["instructions"] == "Which invoice?"
    assert state == "User: my march invoice\n\nKnown about INV-1:\n- INV-1 was paid."
    assert set(questions) == {"urgent"}
    assert out["decision"]["state_text"] == "User: my march invoice"   # routing never sees the facts
    assert "Known about INV-1:\n- INV-1 was paid." in gen.calls[-1]["system"]
    assert "Signals: urgent 80%" in gen.calls[-1]["system"]


def test_a_pick_below_the_threshold_changes_nothing(catalogue_intake):
    router = FakeRouter(decision("Billing"), asks=[pick("INV-1", 0.3)])
    gen = FakeGenerator(NOTHING, "Which invoice?")
    out, _, _ = turn(with_catalogue(catalogue_intake, router, gen), new_session(), "an invoice")

    assert out["draft"] == {}
    assert out["decision"]["catalogue"]["picked"] is False
    assert len(router.ask_calls) == 1
    assert "Known about" not in gen.calls[-1]["system"]


def test_a_low_pick_keeps_a_value_picked_earlier(catalogue_intake):
    router = FakeRouter(decision("Billing"), asks=[pick("INV-1", 0.9), URGENT, pick("INV-2", 0.3)])
    # The extraction gives back the user's own words for the field on every turn
    words = '{"invoice": "the march one", "problem": null}'
    gen = FakeGenerator(words, "What is wrong with it?", words, "What is wrong with it?")
    graph = with_catalogue(catalogue_intake, router, gen)
    _, session, _ = turn(graph, new_session(), "the march one")
    out, _, _ = turn(graph, session, "hmm, let me think")

    assert out["draft"] == {"invoice": "INV-1"}


def test_the_pick_runs_once_in_a_turn_that_calls_tools(catalogue_intake):
    router = FakeRouter(decision("Billing"), asks=[pick("INV-1", 0.9), URGENT])
    gen = FakeGenerator('{"invoice": "INV-1", "problem": "charged twice"}',
                        tool_call("lookup_invoice", invoice_number="INV-1"), "I have looked it up.")
    out, _, _ = turn(with_catalogue(catalogue_intake, router, gen), new_session(), "INV-1 was charged twice")

    assert isinstance(out["messages"][-2], ToolMessage)             # the route's node ran a second time
    assert out["messages"][-1].content == "I have looked it up."
    assert len(router.ask_calls) == 2
    assert gen.calls[-1]["system"].count("Known about INV-1") == 1


def test_no_pick_on_a_clarify_turn(catalogue_intake):
    router = FakeRouter(decision("Billing", confidence=0.3))
    out, _, _ = turn(with_catalogue(catalogue_intake, router, FakeGenerator("Billing or tech?")), new_session(), "hmm")

    assert router.ask_calls == []
    assert "catalogue" not in out["decision"]


def test_routes_outside_the_catalogue_are_untouched(catalogue_intake):
    router = FakeRouter(decision("Tech"))
    turn(with_catalogue(catalogue_intake, router, FakeGenerator("Try restarting it.")), new_session(), "it crashed")

    assert router.ask_calls == []


def test_rules_are_recomputed_from_the_answers_asked_again(write_scenario):
    rules = '\n[[rules]]\nwhen = "urgent >= 0.5"\ntell = "Mark it urgent."\n'
    scenario = load_scenario(write_scenario(INTAKE_TOML + CATALOGUE_TOML + rules))
    router = FakeRouter(decision("Billing"), asks=[pick("INV-1", 0.9), URGENT])
    gen = FakeGenerator(NOTHING, "What is wrong with it?")
    out, _, _ = turn(with_catalogue(scenario, router, gen), new_session(), "the march invoice")

    assert out["decision"]["rules"] == ["urgent >= 0.5"]
    assert "- Mark it urgent." in gen.calls[-1]["system"]


def test_without_entries_the_catalogue_routes_behave_as_before(catalogue_intake):
    router = FakeRouter(decision("Billing"))
    gen = FakeGenerator('{"invoice": "INV-1", "problem": null}', "What is wrong with it?")
    out, _, _ = turn(build_graph(catalogue_intake, router, gen), new_session(), "about INV-1")

    assert out["draft"] == {"invoice": "INV-1"} and router.ask_calls == []


def test_only_the_listed_questions_are_asked_again_with_the_facts(write_scenario):
    two = '\n[questions.angry]\ntype = "noul"\ninstructions = "Is the customer angry?"\n'
    scenario = load_scenario(write_scenario(INTAKE_TOML + two + CATALOGUE_TOML + 'questions = ["angry"]\n'))
    first = {"urgent": {"type": "noul", "noul": 0.1}, "angry": {"type": "noul", "noul": 0.2}}
    angry = {"angry": {"type": "noul", "noul": 0.9}}
    router = FakeRouter(decision("Billing", answers=first), asks=[pick("INV-1", 0.9), angry])
    out, _, _ = turn(with_catalogue(scenario, router, FakeGenerator(NOTHING, "Which problem?")), new_session(), "x")

    assert set(router.ask_calls[1][1]) == {"angry"}
    assert out["decision"]["answers"] == {"urgent": first["urgent"], **angry}


def test_with_no_questions_listed_none_are_asked_again(write_scenario):
    scenario = load_scenario(write_scenario(INTAKE_TOML + CATALOGUE_TOML + "questions = []\n"))
    router = FakeRouter(decision("Billing"), asks=[pick("INV-1", 0.9)])
    out, _, _ = turn(with_catalogue(scenario, router, FakeGenerator(NOTHING, "Which problem?")), new_session(), "x")

    assert len(router.ask_calls) == 1
    assert out["draft"] == {"invoice": "INV-1"}


def test_a_later_confident_pick_replaces_an_earlier_one(catalogue_intake):
    router = FakeRouter(decision("Billing"), asks=[pick("INV-1", 0.9), URGENT, pick("INV-2", 0.9), URGENT])
    gen = FakeGenerator(NOTHING, "What is wrong with it?", NOTHING, "What is wrong with it?")
    graph = with_catalogue(catalogue_intake, router, gen)
    _, session, _ = turn(graph, new_session(), "the march invoice")
    out, _, _ = turn(graph, session, "sorry, I meant April")

    assert out["draft"] == {"invoice": "INV-2"}


def test_thanks_after_a_submission_does_not_open_a_request_even_with_a_confident_pick(catalogue_intake):
    # The conversation CLM reads still holds the submitted request, so its pick on "thanks" is confident
    router = FakeRouter(decision("Billing"), asks=[pick("INV-1", 0.9), URGENT, pick("INV-1", 0.9), URGENT])
    gen = FakeGenerator('{"invoice": "INV-1", "problem": "charged twice"}', submit_call(), "Submitted as F-1.",
                        NOTHING, "You're welcome.")
    graph = with_catalogue(catalogue_intake, router, gen)
    _, session, _ = turn(graph, new_session(), "INV-1 was charged twice")
    assert session["follow_up"] is True

    out, _, _ = turn(graph, session, "thanks!")
    assert out["messages"][-1].content == "You're welcome."
    assert (out["draft"], out["draft_route"], out["intake"]) == ({}, None, None)
