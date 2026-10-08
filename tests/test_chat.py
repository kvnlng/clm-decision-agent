"""Tests for chat.py."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# Fakes mirror real interfaces (so some arguments go unused) and note what they were called with.
# pylint: disable=missing-function-docstring,missing-class-docstring,redefined-outer-name
# pylint: disable=unused-argument,attribute-defined-outside-init,use-implicit-booleaness-not-comparison
import pytest
from conftest import BASE_TOML, CONFIRM_TOML, INTAKE_TOML, FakeGenerator, FakeRouter, decision, tool_call
from test_clm_router import ROUTE_ANSWER, URGENT_ANSWER, FakeEngine, VectorRouter

import chat
from chat import chat_loop, format_data_notice, main, make_recorder
from clm_router import ROUTE_QID, ClmError
from generator import GeneratorError
from graph import build_graph
from record import append_record, load_records, new_key
from scenario import load_scenario


def scripted(lines):
    """A stand-in for input(): returns the lines in order, then raises EOFError like Ctrl-D."""
    remaining = list(lines)

    def read(prompt=""):
        if not remaining:
            raise EOFError
        return remaining.pop(0)
    return read


def run_chat(scenario, router, gen, lines):
    written = []
    chat_loop(build_graph(scenario, router, gen), scenario, read=scripted(lines), write=written.append)
    return "\n".join(written)


def test_turn_prints_decision_then_reply(scenario):
    router = FakeRouter(decision("Billing", 0.9, answers={"urgent": {"type": "noul", "noul": 0.76}}))
    output = run_chat(scenario, router, FakeGenerator("Refund on its way."), ["I was charged twice", "exit"])

    assert "Billing" in output and "90.00%" in output
    assert "confidence 0.90 (threshold 0.60)" in output
    assert "Signals: urgent 76%" in output
    assert "-> Billing" in output
    assert output.rstrip().endswith("agent> Refund on its way.")


def test_low_confidence_is_reported_as_clarify(scenario):
    output = run_chat(scenario, FakeRouter(decision("Billing", 0.3)), FakeGenerator("Which is it?"), ["hm"])
    assert "-> clarify (confidence below threshold)" in output


def test_blank_lines_are_not_sent_to_the_models(scenario):
    router = FakeRouter(decision("Billing"))
    run_chat(scenario, router, FakeGenerator("ok"), ["", "   ", "hello", "exit"])
    assert router.states == ["User: hello"]


def test_history_is_passed_on_the_next_turn(scenario):
    gen = FakeGenerator("first", "second")
    run_chat(scenario, FakeRouter(decision("Billing")), gen, ["one", "two"])
    assert [m.content for m in gen.calls[1]["messages"]] == ["one", "first", "two"]


def test_missing_scenario_exits_with_code_2_before_loading_models(tmp_path, capsys):
    assert main([str(tmp_path / "nope.toml")]) == 2
    assert "scenario file not found" in capsys.readouterr().err


def test_tool_check_is_printed_between_decision_and_reply(scenario):
    gen = FakeGenerator(tool_call("lookup_invoice", invoice_number="INV-2041"), "Charged twice.")
    output = run_chat(scenario, FakeRouter(decision("Billing"), tool_p=0.82), gen, ["check INV-2041"])

    assert '  tool call: lookup_invoice({"invoice_number": "INV-2041"})' in output
    assert "    CLM 82% -> run" in output
    assert output.index("tool call:") < output.index("agent> Charged twice.")


def test_declined_tool_check_says_so(scenario):
    gen = FakeGenerator(tool_call("lookup_invoice", invoice_number="INV-2041"), "Sorry.")
    output = run_chat(scenario, FakeRouter(decision("Billing"), tool_p=0.2), gen, ["check INV-2041"])
    assert "    CLM 20% -> declined" in output


def run_review(scenario, router, gen, lines):
    written = []
    def record(entry):
        append_record(scenario.record_path, {**entry, "scenario_sha": scenario.sha})

    graph = build_graph(scenario, router, gen, review=True, record=record)
    chat_loop(graph, scenario, review=True, read=scripted(lines), write=written.append)
    return "\n".join(written), load_records(scenario.record_path)[0]


def test_review_session_writes_verdicts_to_the_record_file(scenario):
    # reads, in order: you>, route?, acceptable?, note, you>
    output, records = run_review(scenario, FakeRouter(decision("Billing")), FakeGenerator("Try restarting."),
                                 ["my app crashes", "Tech", "y", "good", "exit"])

    assert [r["kind"] for r in records] == ["route", "reply"]
    assert (records[0]["clm"]["route"], records[0]["human"]) == ("Billing", "Tech")
    assert records[1]["human"] == {"acceptable": True, "note": "good"}
    assert all(r["scenario_sha"] == scenario.sha for r in records)
    assert "CLM routing:" in output
    assert "agent> Try restarting." in output
    assert output.count("agent> Try restarting.") == 1


def test_enter_accepts_clm_route_and_prefix_selects_a_route(scenario):
    gen = FakeGenerator("one", "two")
    _, records = run_review(scenario, FakeRouter(decision("Billing")), gen,
                            ["first", "", "y", "", "second", "te", "n", "", "exit"])
    assert [r["human"] for r in records if r["kind"] == "route"] == ["Billing", "Tech"]


def test_unknown_route_answer_is_asked_again(scenario):
    output, records = run_review(scenario, FakeRouter(decision("Billing")), FakeGenerator("ok"),
                                 ["hello", "shipping", "Tech", "y", "", "exit"])
    assert "routes: Billing, Tech" in output
    assert records[0]["human"] == "Tech"


def test_tool_review_prompt_and_record(scenario):
    gen = FakeGenerator(tool_call("lookup_invoice", invoice_number="INV-1"), "Charged twice.")
    output, records = run_review(scenario, FakeRouter(decision("Billing"), tool_p=0.82), gen,
                                 ["check INV-1", "", "y", "y", "", "exit"])
    assert "CLM 82% -> run" in output
    assert [r["kind"] for r in records] == ["route", "tool", "reply"]
    assert records[1]["human"] is True


def test_review_turns_do_not_duplicate_history(scenario):
    gen = FakeGenerator("first", "second")
    run_review(scenario, FakeRouter(decision("Billing")), gen,
               ["one", "", "y", "", "two", "", "y", "", "exit"])
    assert [m.content for m in gen.calls[1]["messages"]] == ["one", "first", "two"]


def test_run_mode_reports_a_matched_correction(scenario):
    corrected = {**decision("Tech", 0.4), "source": "reviewed", "similarity": 0.97, "clm_route": "Billing"}
    output = run_chat(scenario, FakeRouter(corrected), FakeGenerator("Try restarting."), ["my app crashes"])
    assert "-> Tech (reviewed before, similarity 0.97; CLM said Billing)" in output


def route_record(state_text, clm, human):
    return {"kind": "route", "state_text": state_text, "human": human,
            "clm": {"route": clm, "confidence": 0.8, "probabilities": {clm: 0.9}}}


class StubRouter:
    instances = []

    def __init__(self):
        self.loaded = None
        StubRouter.instances.append(self)

    def load_verdicts(self, records, scenario):
        self.loaded = records
        return len(records)


def test_both_modes_load_verdicts(write_scenario, monkeypatch):
    path = write_scenario()
    append_record(path.with_name("scenario.record.jsonl"), route_record("User: a", "Billing", "Tech"))
    seen = []
    monkeypatch.setattr(chat, "make_router", lambda scenario: (StubRouter(), "stub"))
    monkeypatch.setattr(chat, "make_generator", lambda scenario: FakeGenerator())
    monkeypatch.setattr(chat, "chat_loop", lambda graph, scenario, review=False: seen.append(review))
    StubRouter.instances.clear()

    assert main([str(path)]) == 0
    assert main([str(path), "--review"]) == 0
    run_router, review_router = StubRouter.instances[0], StubRouter.instances[1]
    assert len(StubRouter.instances) == 2
    assert [r["human"] for r in run_router.loaded] == ["Tech"]
    assert [r["human"] for r in review_router.loaded] == ["Tech"]
    assert seen == [False, True]


def no_models(*args, **kwargs):
    raise AssertionError("models must not load")


def test_review_with_unwritable_record_exits_2_before_loading_models(write_scenario, monkeypatch, capsys):
    path = write_scenario()
    path.with_name("scenario.record.jsonl").mkdir()
    monkeypatch.setattr(chat, "make_router", no_models)
    assert main([str(path), "--review"]) == 2
    assert "cannot write the record" in capsys.readouterr().err


def test_run_with_unreadable_record_exits_2_before_loading_models(write_scenario, monkeypatch, capsys):
    path = write_scenario()
    path.with_name("scenario.record.jsonl").mkdir()
    monkeypatch.setattr(chat, "make_router", no_models)
    assert main([str(path)]) == 2
    assert "cannot read the record" in capsys.readouterr().err


def test_generator_error_is_reported_and_the_session_continues(scenario):
    class Flaky(FakeGenerator):
        def generate(self, system, messages, tools=None):
            if not self.calls:
                self.calls.append({})
                raise GeneratorError("Cannot reach Ollama.")
            return super().generate(system, messages, tools)

    output = run_chat(scenario, FakeRouter(decision("Tech")), Flaky("ok"), ["one", "two"])
    assert "error: Cannot reach Ollama." in output
    assert output.rstrip().endswith("agent> ok")


def test_review_shows_an_off_route_tool_call_before_the_reply(scenario):
    gen = FakeGenerator(tool_call("issue_refund", invoice_number="INV-1", amount=5.0), "Sorry, cannot.")
    output, _ = run_review(scenario, FakeRouter(decision("Billing")), gen, ["refund me", "", "n", "", "exit"])
    assert "declined: tool not available on this route" in output
    assert output.index("declined: tool not available") < output.index("agent> Sorry, cannot.")


def test_help_lists_routes_and_commands_without_calling_the_models(scenario):
    router = FakeRouter(decision("Billing"))
    output = run_chat(scenario, router, FakeGenerator(), ["help", "?", "exit"])
    assert "Billing" in output and "Charges and refunds" in output
    assert "Tech" in output and "Bugs and outages" in output
    assert "Mode: run" in output
    assert "help" in output and "exit" in output
    assert router.states == []


def test_help_shows_route_examples_when_the_scenario_has_them(write_scenario):
    text = BASE_TOML.replace('system_prompt = "You handle tech."',
                             'system_prompt = "You handle tech."\nexamples = ["The app crashes"]')
    with_examples = load_scenario(write_scenario(text))
    output = run_chat(with_examples, FakeRouter(decision("Billing")), FakeGenerator(), ["help"])
    assert 'e.g. "The app crashes"' in output


def test_review_help_explains_review_mode(scenario):
    output, _ = run_review(scenario, FakeRouter(decision("Billing")), FakeGenerator(), ["help", "exit"])
    assert "Mode: review" in output and "scenario.record.jsonl" in output


def test_route_prompt_announces_a_matching_past_correction(scenario):
    matched = {**decision("Tech", 0.9), "source": "reviewed", "similarity": 0.99, "clm_route": "Billing"}
    output, records = run_review(scenario, FakeRouter(matched), FakeGenerator("Try restarting."),
                                 ["my app crashes", "", "y", "", "exit"])
    assert "reviewed before (similarity 0.99): Tech (CLM alone said Billing)" in output
    assert (records[0]["human"], records[0]["clm"]["route"]) == ("Tech", "Billing")


def test_question_mark_at_the_route_prompt_lists_routes_with_descriptions(scenario):
    output, records = run_review(scenario, FakeRouter(decision("Billing")), FakeGenerator("ok"),
                                 ["hello", "?", "Tech", "y", "", "exit"])
    assert "Tech: Bugs and outages" in output
    assert records[0]["human"] == "Tech"


def billing_engine():
    return FakeEngine({ROUTE_QID: ROUTE_ANSWER, "urgent": URGENT_ANSWER})


def test_recorder_confirms_each_route_verdict_and_reloads_corrections(scenario):
    router = VectorRouter(billing_engine(), {"User: a": [1.0, 0.0]})
    written = []
    record = make_recorder(scenario, router, written.append)
    record(route_record("User: a", "Billing", "Tech"))
    assert written == ["  recorded: Billing -> Tech (1 reviewed route on file)"]
    assert len(router.verdicts) == 1
    record(route_record("User: a", "Billing", "Billing"))
    assert written[-1] == "  recorded: Billing confirmed (1 reviewed route on file)"
    assert [route for _, route in router.verdicts] == ["Billing"]
    assert [r["scenario_sha"] for r in load_records(scenario.record_path)[0]] == [scenario.sha] * 2


def test_a_correction_takes_effect_on_the_same_question_in_the_same_session(scenario):
    again = "User: my app crashes\nAssistant: Try restarting.\nUser: my app crashes"
    router = VectorRouter(billing_engine(), {"User: my app crashes": [1.0, 0.0], again: [0.0, 1.0]})
    written = []
    graph = build_graph(scenario, router, FakeGenerator("Try restarting.", "Still crashing?"), review=True,
                        record=make_recorder(scenario, router, written.append))
    chat_loop(graph, scenario, review=True, write=written.append,
              read=scripted(["my app crashes", "Tech", "y", "", "my app crashes", "", "y", "", "exit"]))
    output = "\n".join(written)

    assert "recorded: Billing -> Tech (1 reviewed route on file)" in output
    assert "reviewed before (similarity 1.00): Tech (CLM alone said Billing)" in output
    routes = [r for r in load_records(scenario.record_path)[0] if r["kind"] == "route"]
    assert [(r["clm"]["route"], r["human"]) for r in routes] == [("Billing", "Tech"), ("Billing", "Tech")]


def test_applied_rules_are_listed_with_the_decision(write_scenario):
    with_rules = load_scenario(write_scenario(
        BASE_TOML + '\n[[rules]]\nwhen = "urgent >= 0.5"\ntell = "Say it is urgent."\n'))
    router = FakeRouter(decision("Tech", answers={"urgent": {"type": "noul", "noul": 0.9}}))
    output = run_chat(with_rules, router, FakeGenerator("On it."), ["it is down"])
    assert "  rules applied: urgent >= 0.5" in output


def test_run_mode_asks_before_a_confirm_tool_and_reports_the_choice(write_scenario):
    scenario = load_scenario(write_scenario(CONFIRM_TOML))
    gen = FakeGenerator(tool_call("issue_refund", invoice_number="INV-1", amount=5.0), "Refunded.")
    output = run_chat(scenario, FakeRouter(decision("Billing"), tool_p=0.82), gen, ["refund INV-1", "y", "exit"])

    assert '  tool call: issue_refund({"amount": 5.0, "invoice_number": "INV-1"})' in output
    assert "    CLM 82% -> run" in output
    assert output.count("tool call: issue_refund") == 1
    assert output.rstrip().endswith("agent> Refunded.")


def intake_scenario(write_scenario):
    return load_scenario(write_scenario(INTAKE_TOML))


def test_the_draft_is_carried_between_turns_and_progress_is_shown(write_scenario):
    scenario = intake_scenario(write_scenario)
    gen = FakeGenerator('{"invoice": "INV-1"}', "What is wrong with it?",
                        '{"problem": "charged twice"}', tool_call("submit_form", summary="INV-1 charged twice"),
                        "Submitted as F-1.")
    output = run_chat(scenario, FakeRouter(decision("Billing")), gen, ["about INV-1", "charged twice", "y", "exit"])

    assert "  fields: 1 of 2 (missing: problem)" in output
    assert "  -> Billing (continuing; CLM said Billing; type 'new' to start over)" in output
    assert "agent> Submitted as F-1." in output


def test_draft_command_shows_what_is_collected(write_scenario):
    scenario = intake_scenario(write_scenario)
    gen = FakeGenerator('{"invoice": "INV-1"}', "What is wrong with it?")
    output = run_chat(scenario, FakeRouter(decision("Billing")), gen, ["about INV-1", "draft", "exit"])
    assert "  Billing draft:\n    Which invoice: INV-1\n    What is wrong with it: (still needed)" in output


def test_draft_command_without_an_open_request(scenario):
    output = run_chat(scenario, FakeRouter(decision("Tech")), FakeGenerator(), ["draft"])
    assert "  no request in progress" in output


def test_new_command_discards_the_draft(write_scenario):
    scenario = intake_scenario(write_scenario)
    router = FakeRouter(decision("Billing"))
    gen = FakeGenerator('{"invoice": "INV-1"}', "What is wrong with it?", '{"invoice": "INV-7"}', "What is wrong?")
    output = run_chat(scenario, router, gen, ["about INV-1", "new", "about INV-7", "draft", "exit"])

    assert "  started over" in output
    assert "Which invoice: INV-7" in output and "INV-1\n    What is wrong" not in output
    assert [m.content for m in gen.calls[2]["messages"]] == ["about INV-7"]


def test_help_lists_the_draft_commands(scenario):
    output = run_chat(scenario, FakeRouter(decision("Tech")), FakeGenerator(), ["help"])
    assert "draft" in output and "new" in output


def test_scenario_path_is_required(capsys):
    with pytest.raises(SystemExit) as stopped:
        main([])
    assert stopped.value.code == 2
    assert "scenario" in capsys.readouterr().err


def test_a_bad_tools_module_exits_2_before_loading_models(write_scenario, monkeypatch, capsys):
    path = write_scenario(BASE_TOML.replace('"fake_tools"', '"no_such_module_xyz"'))
    monkeypatch.setattr(chat, "make_router", no_models)
    assert main([str(path)]) == 2
    assert "cannot import 'no_such_module_xyz'" in capsys.readouterr().err


def test_the_tools_module_is_configured_once_per_start(write_scenario, monkeypatch):
    import fake_tools  # pylint: disable=import-outside-toplevel
    class Router(FakeRouter):
        def load_verdicts(self, records, scenario):
            return 0

    path = write_scenario()
    monkeypatch.setattr(chat, "make_router", lambda scenario: (Router(decision("Tech")), "stub"))
    monkeypatch.setattr(chat, "make_generator", lambda scenario: FakeGenerator())
    monkeypatch.setattr(chat, "chat_loop", lambda *args, **kwargs: None)
    fake_tools.CONFIGURED.clear()
    assert main([str(path)]) == 0
    assert len(fake_tools.CONFIGURED) == 1


def test_data_notice_for_models_in_this_process(scenario):
    notice = format_data_notice(scenario, review=False, encrypted=False)
    assert "CLM: runs in this process" in notice and "language model: runs in this process" in notice
    assert "record: nothing is saved in run mode" in notice


def test_data_notice_says_when_text_leaves_this_computer(write_scenario):
    text = (BASE_TOML.replace('backend = "mlx"', 'backend = "openai"\nbase_url = "https://models.example.org/v1"')
            .replace('model_id = "test/clm"', 'backend = "server"\nurl = "http://localhost:8700"'))
    notice = format_data_notice(load_scenario(write_scenario(text)), review=True, encrypted=True)
    assert "CLM: sent to http://localhost:8700 (on this computer)" in notice
    assert "language model: sent to https://models.example.org/v1 (ANOTHER MACHINE)" in notice
    assert "scenario.record.jsonl (encrypted)" in notice
    plain = format_data_notice(load_scenario(write_scenario(text)), review=True, encrypted=False)
    assert "scenario.record.jsonl (not encrypted)" in plain


def test_review_writes_an_encrypted_private_record_when_the_scenario_names_a_key(write_scenario, monkeypatch):
    key = new_key()
    monkeypatch.setenv("DEMO_RECORD_KEY", key.decode())
    scenario = load_scenario(write_scenario(BASE_TOML + '\n[record]\nkey_env = "DEMO_RECORD_KEY"\n'))
    router = FakeRouter(decision("Billing"), vectors={"User: my app crashes": [1.0, 0.0]})
    router.load_verdicts = lambda records, scenario: len(records)
    make_recorder(scenario, router, write=lambda line: None)(
        {"kind": "route", "state_text": "User: my app crashes", "human": "Tech",
         "clm": {"route": "Billing", "confidence": 0.9, "probabilities": {}}})
    assert b"crashes" not in scenario.record_path.read_bytes()
    assert load_records(scenario.record_path, key=key)[0][0]["human"] == "Tech"


def test_a_missing_record_key_exits_2_before_loading_models(write_scenario, monkeypatch, capsys):
    monkeypatch.delenv("DEMO_MISSING_KEY", raising=False)
    path = write_scenario(BASE_TOML + '\n[record]\nkey_env = "DEMO_MISSING_KEY"\n')
    monkeypatch.setattr(chat, "make_router", no_models)
    assert main([str(path)]) == 2
    assert "DEMO_MISSING_KEY is not set" in capsys.readouterr().err


def test_a_clm_server_error_is_reported_and_the_session_continues(scenario):
    class Flaky(FakeRouter):
        def decide(self, state_text, scenario):
            if not self.states:
                self.states.append(state_text)
                raise ClmError("Cannot reach the CLM server at http://x.")
            return super().decide(state_text, scenario)

    output = run_chat(scenario, Flaky(decision("Tech")), FakeGenerator("ok"), ["one", "two"])
    assert "error: Cannot reach the CLM server at http://x." in output
    assert output.rstrip().endswith("agent> ok")


def test_review_refuses_to_start_when_the_record_needs_another_key(write_scenario, monkeypatch, capsys):
    monkeypatch.setenv("DEMO_RECORD_KEY", new_key().decode())
    path = write_scenario(BASE_TOML + '\n[record]\nkey_env = "DEMO_RECORD_KEY"\n')
    record_path = path.with_name("scenario.record.jsonl")
    append_record(record_path, route_record("User: a", "Billing", "Tech"), key=new_key())
    before = record_path.read_bytes()
    monkeypatch.setattr(chat, "make_router", no_models)
    assert main([str(path), "--review"]) == 2
    assert "1 line of the record cannot be read with the current key setting" in capsys.readouterr().err
    assert record_path.read_bytes() == before


def test_data_notice_warns_when_tracing_would_send_prompts_elsewhere(scenario, monkeypatch):
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    assert "LangSmith" not in format_data_notice(scenario, review=False, encrypted=False)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    assert "ALSO sent to LangSmith" in format_data_notice(scenario, review=False, encrypted=False)
