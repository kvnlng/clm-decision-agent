"""Tests for tools.py."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
import re

import fake_tools
import pytest
from conftest import BASE_TOML

from scenario import ScenarioError, load_scenario
from tools import check_tools, load_tools


def test_load_tools_by_name(write_scenario):
    path = write_scenario(BASE_TOML.replace('module = "fake_tools"',
                                            'module = "fake_tools"\n\n[tools.settings]\nurl = "http://x"'))
    registry = load_tools(load_scenario(path))
    assert registry == {"lookup": fake_tools.lookup, "act": fake_tools.act, "broken": fake_tools.broken,
                        "unserialisable": fake_tools.unserialisable, "slow": fake_tools.slow,
                        "alternatives": fake_tools.alternatives}
    assert fake_tools.SETTINGS == {"url": "http://x", "scenario_dir": str(path.resolve().parent)}


def test_no_module_means_no_tools(write_scenario):
    text = (BASE_TOML.replace('[tools]\nmodule = "fake_tools"\n', "").replace('inform = ["lookup"]\n', "")
            .replace('[tools.inform.lookup]\nquestions = ["urgent"]\n', "").split("[[actions]]", maxsplit=1)[0])
    scenario = load_scenario(write_scenario(text))
    assert load_tools(scenario) == {}
    check_tools(scenario, {})


def test_a_module_beside_the_scenario_is_found(write_scenario, tmp_path):
    (tmp_path / "desk_tools_for_test.py").write_text(
        'def lookup(request):\n    return {"facts": []}\n\ndef act(request):\n    return {}\n\nTOOLS = [lookup, act]\n')
    scenario = load_scenario(write_scenario(BASE_TOML.replace('"fake_tools"', '"desk_tools_for_test"')))
    assert list(load_tools(scenario)) == ["lookup", "act"]


def test_the_scenario_tools_check_passes(scenario):
    check_tools(scenario, load_tools(scenario))


@pytest.mark.parametrize("change, message", [
    (('inform = ["lookup"]', 'inform = ["x"]'), "routes.Billing.inform: unknown tool 'x'"),
    (('tool = "act"', 'tool = "x"'), "actions[0].tool: unknown tool 'x'"),
    (("confidence_threshold = 0.6", 'confidence_threshold = 0.6\ninform = ["x"]'), "routing.inform: unknown tool 'x'"),
    (('[tools.inform.lookup]', '[tools.inform.x]'), "tools.inform.x: unknown tool 'x'"),
    (('[tools]\nmodule = "fake_tools"\n', ""),
     "routes.Billing.inform: unknown tool 'lookup' (no [tools] module is set)"),
])
def test_unknown_tools_are_scenario_errors(write_scenario, change, message):
    bad = load_scenario(write_scenario(BASE_TOML.replace(*change)))
    with pytest.raises(ScenarioError, match=re.escape(message)):
        check_tools(bad, load_tools(bad))


@pytest.mark.parametrize("name, source, message", [
    ("bad_tools_a", "def broken(:\n", "cannot import 'bad_tools_a'"),
    ("bad_tools_b", "TOOLS = [undefined_name]\n", "cannot import 'bad_tools_b'"),
    ("bad_tools_c", "TOOLS = ['lookup']\n", "every entry in TOOLS must be a function"),
    ("bad_tools_e", "x = 1\n", "'bad_tools_e' must define TOOLS, a list of functions"),
    ("bad_tools_d", "from fake_tools import lookup\nTOOLS = [lookup]\n\n"
     "def configure(settings, scenario_dir):\n    raise FileNotFoundError('queues.json')\n",
     "configure() failed: FileNotFoundError: queues.json"),
])
def test_a_broken_tools_module_is_a_scenario_error(write_scenario, tmp_path, name, source, message):
    (tmp_path / f"{name}.py").write_text(source)
    scenario = load_scenario(write_scenario(BASE_TOML.replace('"fake_tools"', f'"{name}"')))
    with pytest.raises(ScenarioError, match=re.escape(message)):
        load_tools(scenario)
