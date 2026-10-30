"""Tests for environment.py: loading an environment's graph, queries, decisions, calculations and scenarios."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
import pytest
from conftest import BASE_TOML, ENV_TOML, limit_dmn

from clm_decision.environment import is_environment, load_environment
from clm_decision.scenario import ScenarioError


def test_loads_the_parts(write_environment):
    env = load_environment(write_environment())
    assert env.name == "Toy sourcing" and env.description == "Where to buy one product's parts"
    assert env.queries["suppliers"].params == {"product": "string"}
    assert env.decisions["share_limit"].params == {"share": "number"}       # from the DMN table's input
    assert env.calculations["cost"].params == {"share": "number", "cost_a": "number", "cost_b": "number"}
    sourcing = env.scenarios["sourcing"]
    assert sourcing.levers == {"share": (0, 0.5, 1.0)} and sourcing.limits == ("share_limit",)
    assert sourcing.objective == "margin" and sourcing.maximise
    assert env.graph.node("S2")["cost"] == 6
    assert env.connector.run(env.queries["suppliers"], {"product": "P1"})[0]["id"] == "S1"


def test_is_environment_tells_the_two_files_apart(write_environment, write_scenario):
    assert is_environment(write_environment())
    assert not is_environment(write_scenario(BASE_TOML))


@pytest.mark.parametrize("old, new, setting", [
    ('edges = "graph/edges.csv"', 'edges = "graph/edges.csv"\nbackend = "memgraph"',
     "graph.backend: must be builtin or neo4j"),
    ('function = "suppliers"', 'function = "nope"', "queries.suppliers.function: 'nope'"),
    ('dmn = "rules/share_limit.dmn"', 'dmn = "rules/missing.dmn"', "decisions.share_limit.dmn: "),
    ('params = { product = "string" }\nfunction', 'params = { product = "text" }\nfunction',
     "queries.suppliers.params.product: "),
    ("levers = { share = [0, 0.5, 1.0] }", "levers = { share = [] }", "scenarios.sourcing.levers.share: "),
    ('limits = ["share_limit"]', 'limits = ["nothing"]', "scenarios.sourcing.limits: unknown decision 'nothing'"),
    ("levers = { share = [0, 0.5, 1.0] }", "levers = {}", "scenarios.sourcing.levers: "),
    ('objective = "margin"', 'objective = 3', "scenarios.sourcing.objective: "),
])
def test_bad_settings_are_one_line(write_environment, old, new, setting):
    assert old in ENV_TOML
    with pytest.raises(ScenarioError, match=f"^{setting}") as caught:
        load_environment(write_environment(ENV_TOML.replace(old, new, 1)))
    assert "\n" not in str(caught.value)


def test_a_functions_module_that_fails_to_import_is_one_line(write_environment):
    with pytest.raises(ScenarioError, match="^environment.functions: cannot import 'toy_functions': ZeroDivision"):
        load_environment(write_environment(functions="1 / 0\n"))


@pytest.mark.parametrize("change, setting", [
    (lambda toml, dmn: (toml, dmn.replace("<text>share</text>", "<text>client.share</text>")),
     "decisions.share_limit.dmn: input 'client.share'"),
    (lambda toml, dmn: (toml.replace('params = { product = "string" }\nfunction',
                                     'params = { from = "string" }\nfunction'), dmn),
     "queries.suppliers.params.from: "),
    (lambda toml, dmn: (toml.replace('params = { product = "string" }\nlevers',
                                     'params = { branches = "list" }\nlevers'), dmn),
     "scenarios.sourcing.params.branches: "),
])
def test_names_the_tools_cannot_take_are_refused(write_environment, change, setting):
    toml, dmn = change(ENV_TOML, limit_dmn())
    with pytest.raises(ScenarioError, match=f"^{setting}"):
        load_environment(write_environment(toml, dmn=dmn))
