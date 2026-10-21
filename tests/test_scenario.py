"""Tests for scenario.py."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# Fakes mirror real interfaces (so some arguments go unused) and note what they were called with.
# pylint: disable=missing-function-docstring,missing-class-docstring,redefined-outer-name
# pylint: disable=unused-argument,attribute-defined-outside-init,use-implicit-booleaness-not-comparison
import re

import pytest

from conftest import BASE_TOML
from scenario import Action, Inform, Rule, ScenarioError, load_scenario


def test_loads_values_and_defaults(write_scenario):
    path = write_scenario()
    s = load_scenario(path)
    assert s.clm_model_id == "test/clm"
    assert (s.generator_backend, s.generator_model, s.max_tokens) == ("mlx", "test/gen", 50)
    assert s.routing_instructions == "Which workflow?"
    assert s.confidence_threshold == 0.6
    assert list(s.routes) == ["Billing", "Tech"]
    assert s.routes["Billing"].description == "Charges and refunds"
    assert s.routes["Billing"].inform == ("lookup",)
    assert s.routes["Tech"].inform == ()
    assert s.questions == {"urgent": {"type": "noul", "instructions": "Is this urgent?"}}
    assert s.tool_check_instructions == "Is this tool call appropriate for what the user asked?"
    assert s.tool_check_threshold == 0.5
    assert (s.live_corrections, s.match_threshold) == (True, 0.97)
    assert s.record_path == path.with_name("scenario.record.jsonl")
    assert len(s.sha) == 64


def test_optional_sections_are_read(write_scenario):
    text = BASE_TOML + '''
[tool_check]
instructions = "OK to run?"
threshold = 0.7

[record]
live_corrections = false
match_threshold = 0.9
'''
    s = load_scenario(write_scenario(text))
    assert (s.tool_check_instructions, s.tool_check_threshold) == ("OK to run?", 0.7)
    assert (s.live_corrections, s.match_threshold) == (False, 0.9)


@pytest.mark.parametrize("old, new, message", [
    ('[routes.Tech]\ndescription = "Bugs and outages"\n', "", "at least two routes"),
    ('description = "Bugs and outages"\n', "", "routes.Tech: missing 'description'"),
    ('backend = "mlx"', 'backend = "cloud"', "generator.backend must be one of mlx, ollama, openai"),
    ('type = "noul"', 'type = "bool"', "questions.urgent.type must be one of noul, choice, score"),
    ("confidence_threshold = 0.6", "confidence_threshold = 1.5", "routing.confidence_threshold"),
    ('instructions = "Which workflow?"\n', "", "routing: missing 'instructions'"),
    ('inform = ["lookup"]', 'inform = "lookup"', "routes.Billing.inform must be a list of tool names"),
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
    text = BASE_TOML.replace('description = "Bugs and outages"',
                             'description = "Bugs and outages"\nexamples = ["The app crashes", "I cannot log in"]')
    s = load_scenario(write_scenario(text))
    assert s.routes["Tech"].examples == ("The app crashes", "I cannot log in")
    assert s.routes["Billing"].examples == ()


def test_route_examples_must_be_a_list_of_strings(write_scenario):
    text = BASE_TOML.replace('description = "Bugs and outages"', 'description = "Bugs and outages"\nexamples = "x"')
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
                             'module = "fake_tools"\n\n[tools.settings]\nurl = "http://x"\nlimit = 3')
    s = load_scenario(write_scenario(text))
    assert s.tools_module == "fake_tools"
    assert s.tool_settings == {"url": "http://x", "limit": 3}


def test_tools_section_is_optional(write_scenario):
    text = BASE_TOML.replace('[tools]\nmodule = "fake_tools"\n', "")
    s = load_scenario(write_scenario(text))
    assert (s.tools_module, s.tool_settings, s.rules) == (None, {}, ())


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


def test_route_inform_and_actions_load(scenario):
    assert scenario.informs == {"lookup": Inform("lookup", ("urgent",))}
    assert scenario.actions == (Action("act", ("Billing",), Rule("urgent >= 0.5", "urgent", ">=", 0.5, ""), False),)


def test_inform_questions_default_to_all(write_scenario):
    s = load_scenario(write_scenario(BASE_TOML.replace('[tools.inform.lookup]\nquestions = ["urgent"]\n', "")))
    assert s.informs == {} and s.routes["Billing"].inform == ("lookup",)
    s = load_scenario(write_scenario(BASE_TOML.replace('questions = ["urgent"]', "")))
    assert s.informs == {"lookup": Inform("lookup", None)}


def test_inform_questions_can_be_none(write_scenario):
    s = load_scenario(write_scenario(BASE_TOML.replace('questions = ["urgent"]', "questions = []")))
    assert s.informs == {"lookup": Inform("lookup", ())}


def test_an_action_without_a_condition_always_applies(write_scenario):
    s = load_scenario(write_scenario(BASE_TOML.replace('when = "urgent >= 0.5"\n', "")))
    assert s.actions[0].when is None


@pytest.mark.parametrize("old, new, message", [
    ('when = "urgent >= 0.5"', 'when = "urgent >= 0.5"\nconfirm = true',
     "actions[0].confirm: only false is supported for now"),
    ('routes = ["Billing"]', 'routes = ["X"]', "actions[0].routes: unknown route 'X'"),
    ('routes = ["Billing"]', 'routes = []', "actions[0].routes must be a non-empty list of route names"),
    ('tool = "act"\n', "", "actions[0]: missing 'tool'"),
    ('when = "urgent >= 0.5"', 'when = "missing >= 1"', "actions[0].when: unknown question 'missing'"),
    ('questions = ["urgent"]', 'questions = ["X"]', "tools.inform.lookup.questions: unknown question 'X'"),
    ('[[actions]]', '[[rules]]\nwhen = "urgent >= 0.5"\ntell = "x"\n\n[actions]', "actions must be written as"),
])
def test_actions_and_informs_name_the_problem(write_scenario, old, new, message):
    with pytest.raises(ScenarioError, match=re.escape(message)):
        load_scenario(write_scenario(BASE_TOML.replace(old, new)))


@pytest.mark.parametrize("old, new, message", [
    ('description = "Bugs and outages"', 'description = "Bugs and outages"\nsystem_prompt = "x"',
     "routes.Tech.system_prompt is no longer used"),
    ('description = "Bugs and outages"', 'description = "Bugs and outages"\ntools = ["act"]',
     "routes.Tech.tools is no longer used"),
    ('description = "Bugs and outages"', 'description = "Bugs and outages"\n\n[routes.Tech.fields]\nx = "y"',
     "routes.Tech.fields is no longer used"),
    ('module = "fake_tools"', 'module = "fake_tools"\nconfirm = ["act"]', "tools.confirm is no longer used"),
    ('[routing]', '[catalogue]\ntool = "x"\n\n[routing]', "[catalogue] is no longer used"),
])
def test_old_keys_are_refused(write_scenario, old, new, message):
    with pytest.raises(ScenarioError, match=re.escape(message)):
        load_scenario(write_scenario(BASE_TOML.replace(old, new)))
