"""Tests for scenario.py."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# Fakes mirror real interfaces (so some arguments go unused) and note what they were called with.
# pylint: disable=missing-function-docstring,missing-class-docstring,redefined-outer-name
# pylint: disable=unused-argument,attribute-defined-outside-init,use-implicit-booleaness-not-comparison
import re

import pytest

from conftest import BASE_TOML, INTAKE_TOML
from scenario import Rule, ScenarioError, load_scenario


def test_loads_values_and_defaults(write_scenario):
    path = write_scenario()
    s = load_scenario(path)
    assert s.clm_model_id == "test/clm"
    assert s.history_turns == 2
    assert (s.generator_backend, s.generator_model, s.max_tokens) == ("mlx", "test/gen", 50)
    assert s.routing_instructions == "Which workflow?"
    assert s.confidence_threshold == 0.6
    assert list(s.routes) == ["Billing", "Tech"]
    assert s.routes["Billing"].description == "Charges and refunds"
    assert s.routes["Billing"].tools == ("lookup_invoice",)
    assert s.routes["Tech"].tools == ()
    assert s.questions == {"urgent": {"type": "noul", "instructions": "Is this urgent?"}}
    assert s.tool_check_instructions == "Is this tool call appropriate for what the user asked?"
    assert (s.tool_check_threshold, s.max_tool_rounds) == (0.5, 3)
    assert (s.live_corrections, s.match_threshold) == (True, 0.97)
    assert s.record_path == path.with_name("scenario.record.jsonl")
    assert len(s.sha) == 64


def test_optional_sections_are_read(write_scenario):
    text = BASE_TOML + '''
[tool_check]
instructions = "OK to run?"
threshold = 0.7
max_tool_rounds = 1

[record]
live_corrections = false
match_threshold = 0.9
'''
    s = load_scenario(write_scenario(text))
    assert (s.tool_check_instructions, s.tool_check_threshold, s.max_tool_rounds) == ("OK to run?", 0.7, 1)
    assert (s.live_corrections, s.match_threshold) == (False, 0.9)


@pytest.mark.parametrize("old, new, message", [
    ('[routes.Tech]\ndescription = "Bugs and outages"\nsystem_prompt = "You handle tech."\n', "",
     "at least two routes"),
    ('system_prompt = "You handle tech."\n', "", "routes.Tech: missing 'system_prompt'"),
    ('backend = "mlx"', 'backend = "cloud"', "generator.backend must be one of mlx, ollama, openai"),
    ('type = "noul"', 'type = "bool"', "questions.urgent.type must be one of noul, choice, score"),
    ("confidence_threshold = 0.6", "confidence_threshold = 1.5", "routing.confidence_threshold"),
    ("[routes.Tech]", "[routes.assess]", "route name 'assess' is reserved"),
    ('instructions = "Which workflow?"\n', "", "routing: missing 'instructions'"),
    ('tools = ["lookup_invoice"]', 'tools = "lookup_invoice"', "routes.Billing.tools must be a list of tool names"),
    ("history_turns = 2", "history_turns = 0", "clm.history_turns"),
])
def test_invalid_scenarios_name_the_problem(write_scenario, old, new, message):
    assert old in BASE_TOML
    with pytest.raises(ScenarioError, match=re.escape(message)):
        load_scenario(write_scenario(BASE_TOML.replace(old, new)))


def test_choice_question_needs_criteria(write_scenario):
    text = BASE_TOML + '\n[questions.team]\ntype = "choice"\ninstructions = "Which team?"\n'
    with pytest.raises(ScenarioError, match="questions.team.criteria"):
        load_scenario(write_scenario(text))


def test_missing_file(tmp_path):
    with pytest.raises(ScenarioError, match="scenario file not found"):
        load_scenario(tmp_path / "nope.toml")


def test_invalid_toml(write_scenario):
    with pytest.raises(ScenarioError, match="invalid TOML"):
        load_scenario(write_scenario("[routing\ninstructions = "))


def test_route_examples_are_optional_and_loaded(write_scenario):
    text = BASE_TOML.replace('system_prompt = "You handle tech."',
                             'system_prompt = "You handle tech."\nexamples = ["The app crashes", "I cannot log in"]')
    s = load_scenario(write_scenario(text))
    assert s.routes["Tech"].examples == ("The app crashes", "I cannot log in")
    assert s.routes["Billing"].examples == ()


def test_route_examples_must_be_a_list_of_strings(write_scenario):
    text = BASE_TOML.replace('system_prompt = "You handle tech."', 'system_prompt = "You handle tech."\nexamples = "x"')
    with pytest.raises(ScenarioError, match="routes.Tech.examples must be a list of example messages"):
        load_scenario(write_scenario(text))

EXTRAS = '''
[questions.level]
type = "score"
instructions = "How bad?"
criteria = ["Low", "Mid", "High"]

[questions.team]
type = "choice"
instructions = "Which team?"
criteria = { a = "Team A", b = "Team B" }
'''


def test_route_fields_keep_file_order(write_scenario):
    text = (INTAKE_TOML.replace("[routes.Billing.fields]", "[routes.Unused.fields]")
            + '\n[routes.Billing.fields]\nsummary = "What is wrong"\nsteps = "How to reproduce it"\n')
    text = text.replace('[routes.Unused.fields]\ninvoice = "Which invoice"\nproblem = "What is wrong with it"\n', "")
    s = load_scenario(write_scenario(text))
    assert list(s.routes["Billing"].fields.items()) == [("summary", "What is wrong"), ("steps", "How to reproduce it")]
    assert s.routes["Tech"].fields == {}


def test_route_fields_must_be_text(write_scenario):
    text = INTAKE_TOML + "summary = 3\n"
    with pytest.raises(ScenarioError, match=re.escape("routes.Billing.fields.summary must be a non-empty string")):
        load_scenario(write_scenario(text))


def test_rules_are_parsed_for_each_question_type(write_scenario):
    text = BASE_TOML + EXTRAS + '''
[[rules]]
when = "urgent >= 0.5"
tell = "Say it is urgent."

[[rules]]
when = "level <= 1"
tell = "Keep it brief."

[[rules]]
when = "team == b"
tell = "Mention team B."
'''
    s = load_scenario(write_scenario(text))
    assert s.rules == (
        Rule("urgent >= 0.5", "urgent", ">=", 0.5, "Say it is urgent."),
        Rule("level <= 1", "level", "<=", 1.0, "Keep it brief."),
        Rule("team == b", "team", "==", "b", "Mention team B."),
    )


@pytest.mark.parametrize("when, message", [
    ("urgent is high", "rules[0].when must look like"),
    ("missing >= 1", "rules[0].when: unknown question 'missing'"),
    ("urgent == yes", "rules[0].when: use >= or <= with a number for question 'urgent'"),
    ("level >= lots", "rules[0].when: use >= or <= with a number for question 'level'"),
    ("team >= 1", "rules[0].when: use == with one of a, b for question 'team'"),
    ("team == c", "rules[0].when: use == with one of a, b for question 'team'"),
])
def test_invalid_rules_name_the_problem(write_scenario, when, message):
    text = BASE_TOML + EXTRAS + f'\n[[rules]]\nwhen = "{when}"\ntell = "x"\n'
    with pytest.raises(ScenarioError, match=re.escape(message)):
        load_scenario(write_scenario(text))


def test_rule_needs_tell(write_scenario):
    text = BASE_TOML + '\n[[rules]]\nwhen = "urgent >= 0.5"\n'
    with pytest.raises(ScenarioError, match=re.escape("rules[0]: missing 'tell'")):
        load_scenario(write_scenario(text))


def test_rule_applies_to_each_answer_type():
    answers = {"urgent": {"type": "noul", "noul": 0.7},
               "level": {"type": "score", "score": 1.4},
               "team": {"type": "choice", "choice": "b"}}
    assert Rule("", "urgent", ">=", 0.5, "").applies(answers)
    assert not Rule("", "urgent", "<=", 0.5, "").applies(answers)
    assert Rule("", "level", "<=", 1.5, "").applies(answers)
    assert Rule("", "team", "==", "b", "").applies(answers)
    assert not Rule("", "team", "==", "a", "").applies(answers)
    assert not Rule("", "absent", ">=", 0.0, "").applies(answers)


def test_matching_rules_returns_those_that_apply(write_scenario):
    text = (BASE_TOML + '\n[[rules]]\nwhen = "urgent >= 0.5"\ntell = "Hurry."\n'
            + '[[rules]]\nwhen = "urgent <= 0.2"\ntell = "Relax."\n')
    s = load_scenario(write_scenario(text))
    assert [r.tell for r in s.matching_rules({"urgent": {"type": "noul", "noul": 0.9}})] == ["Hurry."]


def test_tools_section_is_read(write_scenario):
    text = BASE_TOML.replace('module = "fake_tools"',
                             'module = "fake_tools"\nconfirm = ["issue_refund"]\n\n'
                             '[tools.settings]\nurl = "http://x"\nlimit = 3')
    s = load_scenario(write_scenario(text))
    assert (s.tools_module, s.confirm_tools) == ("fake_tools", ("issue_refund",))
    assert s.tool_settings == {"url": "http://x", "limit": 3}


def test_tools_section_is_optional(write_scenario):
    text = BASE_TOML.replace('[tools]\nmodule = "fake_tools"\n', "")
    s = load_scenario(write_scenario(text))
    assert (s.tools_module, s.confirm_tools, s.tool_settings, s.rules) == (None, (), {}, ())


def test_tools_confirm_must_be_a_list_of_names(write_scenario):
    text = BASE_TOML.replace('module = "fake_tools"', 'module = "fake_tools"\nconfirm = "issue_refund"')
    with pytest.raises(ScenarioError, match=re.escape("tools.confirm must be a list of tool names")):
        load_scenario(write_scenario(text))


def test_the_demo_scenario_loads_and_its_tools_resolve(tmp_path):
    import shutil                              # pylint: disable=import-outside-toplevel
    from pathlib import Path                   # pylint: disable=import-outside-toplevel
    from tools import resolve_tools            # pylint: disable=import-outside-toplevel
    # A copy, because loading the tools builds catalogue.db beside the scenario
    example = Path(__file__).parent.parent / "examples" / "service_desk"
    for name in ("scenario.toml", "queues.json", "catalogue.sql"):
        shutil.copy(example / name, tmp_path / name)
    demo = load_scenario(tmp_path / "scenario.toml")
    assert list(demo.routes) == ["Data_Access", "Research_Support", "Information_Request", "Ticket_Status",
                                 "Out_Of_Scope"]
    assert list(demo.routes["Data_Access"].fields) == ["dataset", "purpose", "detail_level", "duration"]
    assert demo.confirm_tools == ("create_ticket",)
    assert [rule.question for rule in demo.rules] == ["urgency", "identifiable"]
    assert [t.name for t in resolve_tools(demo)["Ticket_Status"]] == ["get_ticket", "search_tickets"]
    assert (tmp_path / "catalogue.db").exists()


def test_a_route_with_fields_needs_a_confirm_tool_to_submit_with(write_scenario):
    text = BASE_TOML + '\n[routes.Billing.fields]\ninvoice = "Which invoice"\n'
    with pytest.raises(ScenarioError, match=re.escape(
            "routes.Billing: a route with fields needs one of its tools listed in tools.confirm")):
        load_scenario(write_scenario(text))


def test_heads_folder_is_resolved_beside_the_scenario(write_scenario):
    path = write_scenario(BASE_TOML.replace('model_id = "test/clm"', 'model_id = "test/clm"\nheads = "my.heads"'))
    assert load_scenario(path).clm_heads == path.parent / "my.heads"
    assert load_scenario(write_scenario()).clm_heads is None


def test_routing_fingerprint_changes_with_the_routing_wording_only(write_scenario):
    base = load_scenario(write_scenario()).routing_sha
    assert load_scenario(write_scenario(BASE_TOML.replace("max_tokens = 50", "max_tokens = 60"))).routing_sha == base
    changes = (("Charges and refunds", "Charges"), ("Which workflow?", "Which one?"), ("routes.Tech", "routes.IT"))
    for old, new in changes:
        assert load_scenario(write_scenario(BASE_TOML.replace(old, new))).routing_sha != base


def test_record_path_can_be_moved(write_scenario, tmp_path):
    s = load_scenario(write_scenario(BASE_TOML + '\n[record]\npath = "private/desk.jsonl"\n'))
    assert s.record_path == tmp_path / "private" / "desk.jsonl"


def test_backends_that_talk_to_a_server_need_an_address(write_scenario):
    text = BASE_TOML.replace('backend = "mlx"', 'backend = "openai"')
    with pytest.raises(ScenarioError, match=re.escape("generator: missing 'base_url'")):
        load_scenario(write_scenario(text))
    text = BASE_TOML.replace('model_id = "test/clm"', 'backend = "server"')
    with pytest.raises(ScenarioError, match=re.escape("clm: missing 'url'")):
        load_scenario(write_scenario(text))


def test_server_backends_are_read(write_scenario):
    text = (BASE_TOML.replace('backend = "mlx"', 'backend = "openai"\nbase_url = "http://localhost:11434/v1"\n'
                                                 'api_key_env = "GEN_KEY"')
            .replace('model_id = "test/clm"', 'backend = "server"\nurl = "http://gpu-box:8700"\nmodel = "clm-latest"'))
    s = load_scenario(write_scenario(text))
    assert (s.generator_backend, s.generator_base_url, s.generator_key_env) == \
        ("openai", "http://localhost:11434/v1", "GEN_KEY")
    assert (s.clm_backend, s.clm_url) == ("server", "http://gpu-box:8700")
    assert (s.clm_key_env, s.clm_model_id) == (None, "clm-latest")


def test_unknown_clm_backend_is_an_error(write_scenario):
    text = BASE_TOML.replace('model_id = "test/clm"', 'backend = "cloud"')
    with pytest.raises(ScenarioError, match=re.escape("clm.backend must be one of mlx, server")):
        load_scenario(write_scenario(text))


@pytest.mark.parametrize("name", ["settle", "confirm_tool"])
def test_names_of_newer_internal_steps_are_reserved_too(write_scenario, name):
    with pytest.raises(ScenarioError, match=re.escape(f"route name '{name}' is reserved")):
        load_scenario(write_scenario(BASE_TOML.replace("routes.Tech", f"routes.{name}")))


@pytest.mark.parametrize("old, new, message", [
    ('model_id = "test/clm"', 'backend = "server"\nurl = "localhost:8700"',
     "clm.url must start with http:// or https://"),
    ('model_id = "test/clm"', 'backend = "server"\nurl = "http://me:pw@host:8700"',
     "clm.url must not contain a user name or password"),
    ('backend = "mlx"', 'backend = "openai"\nbase_url = "http://evil.example\\\\@localhost/v1"',
     "generator.base_url must not contain"),
    ('[routing]', '[record]\nkey_env = "A"\nkey_file = "b.key"\n\n[routing]',
     "record: set key_env or key_file, not both"),
    ('[routing]', '[record]\nkey_file = "~no_such_user_xyz/k"\n\n[routing]', "record.key_file"),
])
def test_addresses_and_key_settings_are_checked(write_scenario, old, new, message):
    with pytest.raises(ScenarioError, match=re.escape(message)):
        load_scenario(write_scenario(BASE_TOML.replace(old, new)))
