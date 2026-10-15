"""Tests for tools.py."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# Fakes mirror real interfaces (so some arguments go unused) and note what they were called with.
# pylint: disable=missing-function-docstring,missing-class-docstring,redefined-outer-name
# pylint: disable=unused-argument,attribute-defined-outside-init,use-implicit-booleaness-not-comparison
import re

import fake_tools
import pytest
from conftest import BASE_TOML, CATALOGUE_ENTRIES, CATALOGUE_TOML, INTAKE_TOML

from scenario import ScenarioError, load_scenario
from tools import load_catalogue, load_tools, resolve_tools


def test_tools_are_loaded_from_the_module_and_configured(write_scenario):
    path = write_scenario(BASE_TOML.replace('module = "fake_tools"',
                                            'module = "fake_tools"\n\n[tools.settings]\nurl = "http://x"'))
    registry = load_tools(load_scenario(path))
    assert sorted(registry) == ["broken_submit", "issue_refund", "list_invoices", "lookup_invoice", "submit_form"]
    assert fake_tools.SETTINGS == {"url": "http://x", "scenario_dir": str(path.resolve().parent)}


def test_no_module_means_no_tools(write_scenario):
    text = BASE_TOML.replace('[tools]\nmodule = "fake_tools"\n', "").replace('tools = ["lookup_invoice"]\n', "")
    assert load_tools(load_scenario(write_scenario(text))) == {}


def test_resolve_tools_maps_each_route(scenario):
    assert resolve_tools(scenario) == {"Billing": [fake_tools.lookup_invoice], "Tech": []}


def test_a_module_beside_the_scenario_is_found(write_scenario, tmp_path):
    (tmp_path / "desk_tools_for_test.py").write_text(
        'from langchain_core.tools import tool\n\n@tool\ndef lookup_invoice(invoice_number: str) -> dict:\n'
        '    """Look up."""\n    return {}\n\nTOOLS = [lookup_invoice]\n')
    scenario = load_scenario(write_scenario(BASE_TOML.replace('"fake_tools"', '"desk_tools_for_test"')))
    assert list(load_tools(scenario)) == ["lookup_invoice"]


@pytest.mark.parametrize("change, message", [
    (('"fake_tools"', '"no_such_module_xyz"'), "tools.module: cannot import 'no_such_module_xyz'"),
    (('"fake_tools"', '"json"'), "tools.module: 'json' must define TOOLS, a list of tools"),
    (('"lookup_invoice"', '"send_email"'), "routes.Billing.tools: unknown tool 'send_email'"),
    (('module = "fake_tools"', 'module = "fake_tools"\nconfirm = ["send_email"]'),
     "tools.confirm: unknown tool 'send_email'"),
    (('[tools]\nmodule = "fake_tools"\n', ""),
     "routes.Billing.tools: unknown tool 'lookup_invoice' (no [tools] module is set)"),
])
def test_tool_problems_are_scenario_errors(write_scenario, change, message):
    bad = load_scenario(write_scenario(BASE_TOML.replace(*change)))
    with pytest.raises(ScenarioError, match=re.escape(message)):
        resolve_tools(bad)


@pytest.mark.parametrize("name, source, message", [
    ("bad_tools_a", "def broken(:\n", "cannot import 'bad_tools_a'"),
    ("bad_tools_b", "TOOLS = [undefined_name]\n", "cannot import 'bad_tools_b'"),
    ("bad_tools_c", "def plain():\n    return 1\n\nTOOLS = [plain]\n", "every entry in TOOLS must be a tool"),
    ("bad_tools_d", "from fake_tools import lookup_invoice\nTOOLS = [lookup_invoice]\n\n"
     "def configure(settings, scenario_dir):\n    raise FileNotFoundError('queues.json')\n",
     "configure() failed: FileNotFoundError: queues.json"),
])
def test_a_broken_tools_module_is_a_scenario_error(write_scenario, tmp_path, name, source, message):
    (tmp_path / f"{name}.py").write_text(source)
    scenario = load_scenario(write_scenario(BASE_TOML.replace('"fake_tools"', f'"{name}"')))
    with pytest.raises(ScenarioError, match=re.escape(message)):
        load_tools(scenario)


@pytest.fixture
def with_catalogue(write_scenario, monkeypatch):
    def _load(entries):
        monkeypatch.setattr(fake_tools, "CATALOGUE", entries)
        loaded = load_scenario(write_scenario(INTAKE_TOML + CATALOGUE_TOML))
        return load_catalogue(loaded, load_tools(loaded))
    return _load


def test_no_catalogue_means_no_call(scenario):
    assert load_catalogue(scenario, load_tools(scenario)) is None


def test_catalogue_entries_are_returned_as_given(with_catalogue):
    assert with_catalogue(CATALOGUE_ENTRIES) == CATALOGUE_ENTRIES


GOOD, OTHER = CATALOGUE_ENTRIES


@pytest.mark.parametrize("entries, message", [
    ({"name": "x"}, "must return a list of entries"),
    ([GOOD], "returned 1 entry; it needs from 2 to 50"),
    ([dict(GOOD, name=f"N{i}") for i in range(51)], "returned 51 entries; it needs from 2 to 50"),
    ([GOOD, "INV-2"], "entry 2 must be a table of name, description and facts"),
    ([GOOD, dict(OTHER, name="INV-1")], "entry 2: the name 'INV-1' is used twice"),
    ([GOOD, dict(OTHER, description=" ")], "entry 2: 'description' must be non-empty text"),
    ([GOOD, dict(OTHER, facts="paid")], "entry 2: 'facts' must be a list of sentences"),
    ([GOOD, {"name": "INV-2"}], "entry 2: 'description' must be non-empty text"),
    (["fail"], "The billing system could not be reached."),
])
def test_catalogue_contract_errors_are_one_line(with_catalogue, entries, message):
    with pytest.raises(ScenarioError, match=re.escape("catalogue.tool 'list_invoices'") + ".*" + re.escape(message)):
        with_catalogue(entries)


def test_an_unknown_catalogue_tool_is_a_scenario_error(write_scenario):
    loaded = load_scenario(write_scenario(INTAKE_TOML + CATALOGUE_TOML.replace("list_invoices", "nope")))
    with pytest.raises(ScenarioError, match=re.escape("catalogue.tool: unknown tool 'nope'")):
        load_catalogue(loaded, load_tools(loaded))
