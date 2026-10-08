"""Tests for graph.py."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# Fakes mirror real interfaces (so some arguments go unused) and note what they were called with.
# pylint: disable=missing-function-docstring,missing-class-docstring,redefined-outer-name
# pylint: disable=unused-argument,attribute-defined-outside-init,use-implicit-booleaness-not-comparison
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from conftest import BASE_TOML, CONFIRM_TOML, FakeGenerator, FakeRouter, decision, tool_call
from graph import DECLINED, USER_DECLINED, build_graph, guidance_block, render_state, signals_line
from scenario import load_scenario


def ask(graph, text):
    return graph.invoke({"messages": [HumanMessage(content=text)]})


def test_confident_decision_goes_to_the_route(scenario):
    router = FakeRouter(decision("Billing", 0.9, answers={"urgent": {"type": "noul", "noul": 0.76}}))
    gen = FakeGenerator("Refund on its way.")
    out = ask(build_graph(scenario, router, gen), "I was charged twice")

    assert out["messages"][-1].content == "Refund on its way."
    assert gen.calls[0]["system"] == "You handle billing.\n\nSignals: urgent 76%"
    assert router.states == ["User: I was charged twice"]
    assert out["decision"]["route"] == "Billing"
    assert out["decision"]["state_text"] == "User: I was charged twice"


def test_low_confidence_goes_to_clarify(scenario):
    gen = FakeGenerator("Is this about a charge or a bug?")
    out = ask(build_graph(scenario, FakeRouter(decision("Billing", 0.3)), gen), "it is broken")

    assert out["messages"][-1].content == "Is this about a charge or a bug?"
    system = gen.calls[0]["system"]
    assert "- Billing: Charges and refunds" in system
    assert "- Tech: Bugs and outages" in system
    assert "You handle billing." not in system


def test_confidence_equal_to_threshold_goes_to_the_route(scenario):
    gen = FakeGenerator("ok")
    ask(build_graph(scenario, FakeRouter(decision("Tech", 0.6)), gen), "hi")
    assert gen.calls[0]["system"] == "You handle tech."


def test_render_state_keeps_the_last_turns_and_skips_tool_traffic():
    messages = [
        HumanMessage(content="one"), AIMessage(content="reply one"),
        HumanMessage(content="two"),
        AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "c"}]),
        ToolMessage(content="{}", tool_call_id="c", name="t"),
        AIMessage(content="reply two"),
        HumanMessage(content="three"),
    ]
    assert render_state(messages, 2) == "User: two\nAssistant: reply two\nUser: three"


def test_signals_line_formats_each_answer_type():
    answers = {
        "urgent": {"type": "noul", "noul": 0.761},
        "frustration": {"type": "score", "score": 2.0, "probabilities": {"0": 0, "1": 0, "2": 1}},
        "team": {"type": "choice", "choice": "billing"},
    }
    assert signals_line(answers) == "\n\nSignals: urgent 76%, frustration 2.0/2, team billing"
    assert signals_line({}) == ""


def test_approved_tool_call_runs_and_returns_to_the_route(scenario):
    router = FakeRouter(decision("Billing"), tool_p=0.9)
    gen = FakeGenerator(tool_call("lookup_invoice", invoice_number="INV-2041"), "It was charged twice.")
    out = ask(build_graph(scenario, router, gen), "check INV-2041")

    kinds = [type(m).__name__ for m in out["messages"]]
    assert kinds == ["HumanMessage", "AIMessage", "ToolMessage", "AIMessage"]
    assert "INV-2041" in out["messages"][2].content
    assert out["messages"][-1].content == "It was charged twice."
    assert [t.name for t in gen.calls[0]["tools"]] == ["lookup_invoice"]
    assert router.tool_states[0] == (
        'User: check INV-2041\n\nProposed tool call:\n- lookup_invoice({"invoice_number": "INV-2041"})')
    check = out["tool_checks"][0]
    assert (check["approved"], check["p"], check["source"]) == (True, 0.9, "clm")
    assert check["calls"] == [{"name": "lookup_invoice", "args": {"invoice_number": "INV-2041"}}]
    assert out["tool_rounds"] == 1


def test_declined_tool_call_is_answered_without_the_tool(scenario):
    router = FakeRouter(decision("Billing"), tool_p=0.1)
    gen = FakeGenerator(tool_call("lookup_invoice", invoice_number="INV-2041"), "I cannot look that up.")
    out = ask(build_graph(scenario, router, gen), "check INV-2041")

    declined = out["messages"][2]
    assert isinstance(declined, ToolMessage)
    assert (declined.content, declined.tool_call_id, declined.name) == (DECLINED, "call_1", "lookup_invoice")
    assert out["messages"][-1].content == "I cannot look that up."
    assert out["tool_checks"][0]["approved"] is False


def test_tool_not_on_the_route_is_declined_without_asking_clm(scenario):
    router = FakeRouter(decision("Billing"), tool_p=0.9)
    gen = FakeGenerator(tool_call("issue_refund", invoice_number="INV-1", amount=5.0), "Sorry.")
    out = ask(build_graph(scenario, router, gen), "refund me")

    assert router.tool_states == []
    assert out["tool_checks"][0]["source"] == "unknown_tool"
    assert out["messages"][2].content == DECLINED
    assert out["messages"][-1].content == "Sorry."


def test_wrong_argument_names_come_back_as_an_error_message_not_a_crash(scenario):
    gen = FakeGenerator(tool_call("lookup_invoice", number="INV-1"), "Which invoice number?")
    out = ask(build_graph(scenario, FakeRouter(decision("Billing")), gen), "check it")

    assert out["messages"][2].status == "error"
    assert out["messages"][-1].content == "Which invoice number?"


def test_tools_are_withdrawn_after_max_tool_rounds(write_scenario):
    limited = load_scenario(write_scenario(BASE_TOML + "\n[tool_check]\nmax_tool_rounds = 1\n"))
    gen = FakeGenerator(tool_call("lookup_invoice", invoice_number="INV-1"), "Done.")
    out = ask(build_graph(limited, FakeRouter(decision("Billing")), gen), "check INV-1")

    assert gen.calls[0]["tools"] is not None
    assert gen.calls[1]["tools"] is None
    assert out["messages"][-1].content == "Done."


def test_route_without_tools_is_offered_none(scenario):
    gen = FakeGenerator("Try restarting.")
    out = ask(build_graph(scenario, FakeRouter(decision("Tech")), gen), "it crashes")
    assert gen.calls[0]["tools"] is None
    assert out["tool_checks"] == []


def test_tool_counters_reset_each_turn(scenario):
    router = FakeRouter(decision("Billing"))
    gen = FakeGenerator(tool_call("lookup_invoice", invoice_number="INV-1"), "First.", "Second.")
    graph = build_graph(scenario, router, gen)
    first = ask(graph, "check INV-1")
    second = graph.invoke({"messages": first["messages"] + [HumanMessage(content="thanks")]})
    assert (second["tool_rounds"], second["tool_checks"]) == (0, [])


def review_turn(graph, text, answers):
    """Run one turn in review mode, answering each pause from `answers` (keyed by payload kind)."""
    config = {"configurable": {"thread_id": "t"}}
    out = graph.invoke({"messages": [HumanMessage(content=text)]}, config)
    kinds = []
    while out.get("__interrupt__"):
        payload = out["__interrupt__"][0].value
        kinds.append(payload["kind"])
        out = graph.invoke(Command(resume=answers[payload["kind"]]), config)
    return out, kinds


def test_review_corrects_the_route_and_records_both_verdicts(scenario):
    records = []
    gen = FakeGenerator("Try restarting.")
    graph = build_graph(scenario, FakeRouter(decision("Billing")), gen, review=True, record=records.append)
    out, kinds = review_turn(graph, "my app crashes", {
        "route": {"route": "Tech"}, "reply": {"acceptable": True, "note": "fine"}})

    assert kinds == ["route", "reply"]
    assert gen.calls[0]["system"] == "You handle tech."
    assert (out["decision"]["route"], out["decision"]["source"], out["decision"]["clm_route"]) == \
        ("Tech", "human", "Billing")
    assert records == [
        {"kind": "route", "state_text": "User: my app crashes", "human": "Tech",
         "clm": {"route": "Billing", "confidence": 0.9, "probabilities": {"Billing": 0.9, "Tech": 0.1}}},
        {"kind": "reply", "state_text": "User: my app crashes", "route": "Tech",
         "reply": "Try restarting.", "off_route_calls": [], "human": {"acceptable": True, "note": "fine"}},
    ]


def test_reviewed_route_is_used_even_when_clm_was_unsure(scenario):
    gen = FakeGenerator("Refund issued.")
    graph = build_graph(scenario, FakeRouter(decision("Billing", 0.2)), gen, review=True, record=lambda e: None)
    review_turn(graph, "hm", {"route": {"route": "Billing"}, "reply": {"acceptable": True, "note": ""}})
    assert gen.calls[0]["system"] == "You handle billing."


def test_reviewer_can_decline_a_tool_call_clm_approved(scenario):
    records = []
    gen = FakeGenerator(tool_call("lookup_invoice", invoice_number="INV-1"), "I will not look that up.")
    graph = build_graph(scenario, FakeRouter(decision("Billing"), tool_p=0.9), gen,
                        review=True, record=records.append)
    out, kinds = review_turn(graph, "check INV-1", {
        "route": {"route": "Billing"}, "tool": {"approved": False},
        "reply": {"acceptable": False, "note": "should have looked"}})

    assert kinds == ["route", "tool", "reply"]
    assert out["messages"][2].content == DECLINED
    check = out["tool_checks"][0]
    assert (check["approved"], check["source"], check["clm_approved"]) == (False, "human", True)
    tool_record = records[1]
    assert tool_record["kind"] == "tool"
    assert tool_record["human"] is False
    assert tool_record["clm"] == {"p": 0.9, "approved": True}
    assert tool_record["calls"] == [{"name": "lookup_invoice", "args": {"invoice_number": "INV-1"}}]
    assert tool_record["state_text"].endswith('- lookup_invoice({"invoice_number": "INV-1"})')
    assert records[2]["human"] == {"acceptable": False, "note": "should have looked"}


def test_reviewer_can_approve_a_tool_call_clm_declined(scenario):
    gen = FakeGenerator(tool_call("lookup_invoice", invoice_number="INV-1"), "Charged twice.")
    graph = build_graph(scenario, FakeRouter(decision("Billing"), tool_p=0.1), gen,
                        review=True, record=lambda e: None)
    out, _ = review_turn(graph, "check INV-1", {
        "route": {"route": "Billing"}, "tool": {"approved": True}, "reply": {"acceptable": True, "note": ""}})
    assert "INV-1" in out["messages"][2].content
    assert out["messages"][2].content != DECLINED


def test_run_mode_never_pauses_and_never_records(scenario):
    records = []
    graph = build_graph(scenario, FakeRouter(decision("Billing")), FakeGenerator("ok"), record=records.append)
    out = ask(graph, "hello")
    assert "__interrupt__" not in out
    assert records == []


def test_a_matched_correction_skips_the_confidence_threshold(scenario):
    corrected = {**decision("Tech", 0.1), "source": "reviewed", "similarity": 0.97, "clm_route": "Billing"}
    gen = FakeGenerator("Try restarting.")
    ask(build_graph(scenario, FakeRouter(corrected), gen), "my app crashes")
    assert gen.calls[0]["system"] == "You handle tech."


def tool_history():
    return [
        HumanMessage(content="check INV-1"),
        AIMessage(content="", tool_calls=[{"name": "lookup_invoice", "args": {"invoice_number": "INV-1"}, "id": "c1"}]),
        ToolMessage(content="{}", tool_call_id="c1", name="lookup_invoice"),
        AIMessage(content="Charged twice."),
    ]


def test_route_without_tools_does_not_see_earlier_tool_traffic(scenario):
    gen = FakeGenerator("Try restarting.")
    graph = build_graph(scenario, FakeRouter(decision("Tech")), gen)
    graph.invoke({"messages": tool_history() + [HumanMessage(content="my app crashes")]})
    assert [(type(m).__name__, m.content) for m in gen.calls[0]["messages"]] == [
        ("HumanMessage", "check INV-1"), ("AIMessage", "Charged twice."), ("HumanMessage", "my app crashes")]


def test_clarify_does_not_see_earlier_tool_traffic(scenario):
    gen = FakeGenerator("Which is it?")
    graph = build_graph(scenario, FakeRouter(decision("Tech", 0.2)), gen)
    graph.invoke({"messages": tool_history() + [HumanMessage(content="hm")]})
    assert [type(m).__name__ for m in gen.calls[0]["messages"]] == ["HumanMessage", "AIMessage", "HumanMessage"]


def test_declined_message_tells_the_generator_not_to_invent_the_result():
    assert "no data" in DECLINED and "Do not" in DECLINED


def test_off_route_tool_call_is_shown_to_the_reviewer_and_recorded(scenario):
    records = []
    gen = FakeGenerator(tool_call("issue_refund", invoice_number="INV-1", amount=5.0), "Sorry, cannot.")
    graph = build_graph(scenario, FakeRouter(decision("Billing")), gen, review=True, record=records.append)
    config = {"configurable": {"thread_id": "t"}}
    out = graph.invoke({"messages": [HumanMessage(content="refund me")]}, config)
    out = graph.invoke(Command(resume={"route": "Billing"}), config)
    payload = out["__interrupt__"][0].value
    assert payload["kind"] == "reply"
    assert [c["source"] for c in payload["tool_checks"]] == ["unknown_tool"]
    graph.invoke(Command(resume={"acceptable": False, "note": ""}), config)
    assert records[-1]["off_route_calls"] == [
        {"name": "issue_refund", "args": {"invoice_number": "INV-1", "amount": 5.0}}]


def test_review_records_clm_own_route_when_a_correction_matched(scenario):
    records = []
    matched = {**decision("Tech", 0.9), "source": "reviewed", "similarity": 0.99, "clm_route": "Billing"}
    graph = build_graph(scenario, FakeRouter(matched), FakeGenerator("Try restarting."),
                        review=True, record=records.append)
    out, _ = review_turn(graph, "my app crashes", {
        "route": {"route": "Tech"}, "reply": {"acceptable": True, "note": ""}})
    assert records[0]["clm"]["route"] == "Billing"
    assert records[0]["human"] == "Tech"
    assert out["decision"]["clm_route"] == "Billing"


RULES = '''
[[rules]]
when = "urgent >= 0.5"
tell = "Say it is urgent."

[[rules]]
when = "urgent <= 0.2"
tell = "Say there is no rush."
'''


def test_matching_rule_is_added_to_the_route_prompt_as_guidance(write_scenario):
    with_rules = load_scenario(write_scenario(BASE_TOML + RULES))
    router = FakeRouter(decision("Tech", answers={"urgent": {"type": "noul", "noul": 0.9}}))
    gen = FakeGenerator("On it.")
    out = ask(build_graph(with_rules, router, gen), "it is down")
    assert gen.calls[0]["system"] == "You handle tech.\n\nSignals: urgent 90%\n\nGuidance:\n- Say it is urgent."
    assert out["decision"]["rules"] == ["urgent >= 0.5"]


def test_no_matching_rule_adds_nothing(write_scenario):
    with_rules = load_scenario(write_scenario(BASE_TOML + RULES))
    assert guidance_block(with_rules, {"urgent": {"type": "noul", "noul": 0.35}}) == ""


def run_with_confirm(graph, text, approved):
    """One run-mode turn; answers every confirmation pause with `approved`."""
    config = {"configurable": {"thread_id": uuid4().hex}}
    out = graph.invoke({"messages": [HumanMessage(content=text)]}, config)
    kinds = []
    while out.get("__interrupt__"):
        kinds.append(out["__interrupt__"][0].value["kind"])
        out = graph.invoke(Command(resume={"approved": approved}), config)
    return out, kinds


def refund_call():
    return tool_call("issue_refund", invoice_number="INV-1", amount=5.0)


def test_confirm_tool_runs_only_after_the_user_says_yes(write_scenario):
    scenario = load_scenario(write_scenario(CONFIRM_TOML))
    gen = FakeGenerator(refund_call(), "Refunded.")
    out, kinds = run_with_confirm(build_graph(scenario, FakeRouter(decision("Billing")), gen), "refund INV-1", True)

    assert kinds == ["confirm"]
    assert "refund issued" in out["messages"][2].content
    check = out["tool_checks"][0]
    assert (check["approved"], check["source"], check["clm_approved"]) == (True, "user", True)


def test_refusing_confirmation_runs_nothing_and_tells_the_generator(write_scenario):
    scenario = load_scenario(write_scenario(CONFIRM_TOML))
    gen = FakeGenerator(refund_call(), "What would you like to change?")
    out, _ = run_with_confirm(build_graph(scenario, FakeRouter(decision("Billing")), gen), "refund INV-1", False)

    assert out["messages"][2].content == USER_DECLINED
    assert out["messages"][-1].content == "What would you like to change?"
    assert out["tool_checks"][0]["approved"] is False


def test_the_user_is_asked_even_when_clm_would_decline(write_scenario):
    scenario = load_scenario(write_scenario(CONFIRM_TOML))
    gen = FakeGenerator(refund_call(), "Refunded.")
    graph = build_graph(scenario, FakeRouter(decision("Billing"), tool_p=0.1), gen)
    out, kinds = run_with_confirm(graph, "refund INV-1", True)

    assert kinds == ["confirm"]
    assert "refund issued" in out["messages"][2].content
    assert out["tool_checks"][0]["clm_approved"] is False


def test_tools_not_marked_confirm_do_not_pause(write_scenario):
    scenario = load_scenario(write_scenario(CONFIRM_TOML))
    gen = FakeGenerator(tool_call("lookup_invoice", invoice_number="INV-1"), "Charged twice.")
    _, kinds = run_with_confirm(build_graph(scenario, FakeRouter(decision("Billing")), gen), "check INV-1", True)
    assert kinds == []


def test_review_mode_asks_once_for_a_confirm_tool(write_scenario):
    scenario = load_scenario(write_scenario(CONFIRM_TOML))
    gen = FakeGenerator(refund_call(), "What would you like to change?")
    graph = build_graph(scenario, FakeRouter(decision("Billing")), gen, review=True, record=lambda e: None)
    out, kinds = review_turn(graph, "refund INV-1", {
        "route": {"route": "Billing"}, "tool": {"approved": False}, "reply": {"acceptable": True, "note": ""}})

    assert kinds == ["route", "tool", "reply"]
    assert out["messages"][2].content == USER_DECLINED
