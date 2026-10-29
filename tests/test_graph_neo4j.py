"""Tests for graph_neo4j.py and load_graph.py: named queries on Neo4j, and copying a CSV graph into it."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# Fakes mirror the neo4j driver's interface, so some arguments go unused.
# pylint: disable=missing-function-docstring,missing-class-docstring,redefined-outer-name,unused-argument
import os
from pathlib import Path

import pytest
from conftest import ENV_TOML

import load_graph
from engine import StepError, run_query, run_scenario
from environment import Query, load_environment
from graph_neo4j import Neo4jConnector
from scenario import ScenarioError

EXAMPLE = Path(__file__).parent.parent / "examples" / "tariff" / "environment.toml"
NEO4J = '''
[graph.neo4j]
uri = "bolt://127.0.0.1:1"
password_env = "TOY_PASSWORD"
'''
CYPHER = '''function = "suppliers"
cypher = "MATCH (s:Supplier)-[:SUPPLIES]->(:Node {id: $product}) RETURN s.id AS id"'''


def neo4j_toml(cypher=True):
    toml = ENV_TOML.replace('edges = "graph/edges.csv"', 'edges = "graph/edges.csv"\nbackend = "neo4j"') + NEO4J
    return toml.replace('function = "suppliers"', CYPHER) if cypher else toml


class FakeSession:
    def __init__(self, driver):
        self.driver = driver

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, cypher, parameters=None, **keywords):
        self.driver.ran.append((cypher, {**(parameters or {}), **keywords}))
        if self.driver.fail:
            raise RuntimeError("Neo.ClientError.Statement.SyntaxError: bad query")
        return [type("Record", (), {"data": lambda self, row=row: row})() for row in self.driver.rows]


class FakeDriver:
    def __init__(self, rows=(), fail=False):
        self.rows, self.fail, self.ran, self.databases = list(rows), fail, [], []

    def session(self, database=None):
        self.databases.append(database)
        return FakeSession(self)


def test_neo4j_backend_needs_cypher(write_environment):
    with pytest.raises(ScenarioError, match="^queries.suppliers.cypher: the neo4j backend runs"):
        load_environment(write_environment(neo4j_toml(cypher=False)))


def test_password_comes_from_the_named_variable(write_environment, monkeypatch):
    monkeypatch.delenv("TOY_PASSWORD", raising=False)
    with pytest.raises(ScenarioError, match="^graph.neo4j.password_env: set TOY_PASSWORD"):
        load_environment(write_environment(neo4j_toml()))


def test_an_unreachable_database_is_one_line(write_environment, monkeypatch):
    pytest.importorskip("neo4j")
    monkeypatch.setenv("TOY_PASSWORD", "not-a-real-password")
    with pytest.raises(ScenarioError, match=r"^graph.neo4j: cannot connect to bolt://127\.0\.0\.1:1") as caught:
        load_environment(write_environment(neo4j_toml()))
    assert "\n" not in str(caught.value)


def test_the_connector_returns_record_data():
    driver = FakeDriver(rows=[{"id": "S1"}, {"id": "S2"}])
    connector = Neo4jConnector(driver, "bolt://test", "sales")
    query = Query("suppliers", "", {"product": "string"}, None, "MATCH (s) RETURN s.id AS id")
    assert connector.run(query, {"product": "P1"}) == [{"id": "S1"}, {"id": "S2"}]
    assert driver.ran == [("MATCH (s) RETURN s.id AS id", {"product": "P1"})] and driver.databases == ["sales"]


def test_a_failing_query_is_a_step_error():
    connector = Neo4jConnector(FakeDriver(fail=True), "bolt://test", "neo4j", errors=(RuntimeError,))
    with pytest.raises(StepError, match="^query suppliers: Neo4j at bolt://test: .*SyntaxError"):
        connector.run(Query("suppliers", "", {}, None, "MATCH"), {})


def test_load_graph_writes_nodes_and_edges():
    env = load_environment(EXAMPLE)
    driver = FakeDriver()
    counts = load_graph.load(env.graph, driver, "neo4j")
    assert counts == {"nodes": 20, "edges": 24}
    statements = [cypher for cypher, _ in driver.ran]
    assert statements[0] == "MATCH (n) DETACH DELETE n" and "REQUIRE n.id IS UNIQUE" in statements[1]
    providers = next(parameters for cypher, parameters in driver.ran if ":Node:Provider" in cypher)
    assert {row["name"] for row in providers["rows"]} >= {"Kestrel Polymers"}
    depends = next(parameters for cypher, parameters in driver.ran if ":DEPENDS_ON]" in cypher)
    assert {"source": "SB1", "target": "RM4", "properties": {}} in depends["rows"]
    supplies = next(parameters for cypher, parameters in driver.ran if ":SUPPLIES]" in cypher)
    assert {"source": "SA1", "target": "C1", "properties": {"unit_price": 30}} in supplies["rows"]


def test_a_kind_that_cannot_be_a_label_is_refused(tmp_path):
    (tmp_path / "nodes.csv").write_text("id,kind,name\nA,bad kind!,A\n", encoding="utf-8")
    (tmp_path / "edges.csv").write_text("source,relation,target\n", encoding="utf-8")
    from graph import Graph  # pylint: disable=import-outside-toplevel
    with pytest.raises(ValueError, match="kind 'bad kind!' cannot be a Neo4j label"):
        load_graph.load(Graph.from_csv(tmp_path / "nodes.csv", tmp_path / "edges.csv"), FakeDriver(), "neo4j")


def _sorted(rows):
    return sorted(rows, key=repr)


@pytest.mark.slow
def test_the_two_graphs_agree():
    """With a local Neo4j (NEO4J_PASSWORD set): load the example, then compare every query and every branch."""
    if not os.environ.get("NEO4J_PASSWORD"):
        pytest.skip("set NEO4J_PASSWORD (see .neo4j-local.env) to run against a local Neo4j")
    builtin = load_environment(EXAMPLE)
    assert load_graph.main([str(EXAMPLE)]) == 0
    neo4j = load_environment(EXAMPLE, backend="neo4j")
    suppliers = ["SA1", "SA2", "SA3", "SB1", "SB2"]
    calls = [("product_facts", {"product": "PX"}), ("components_of", {"product": "PX"}),
             ("suppliers_of", {"component": "C1"}), ("suppliers_of", {"component": "C3"}),
             ("key_clients_of", {"product": "PX"}), ("supply_dependencies", {"suppliers": suppliers}),
             ("single_points_of_failure", {"suppliers": ["SB1", "SB2"]}),
             ("single_points_of_failure", {"suppliers": suppliers}), ("single_points_of_failure", {"suppliers": []})]
    for name, params in calls:
        ours = _sorted(run_query(builtin, name, params)["rows"])
        assert _sorted(run_query(neo4j, name, params)["rows"]) == ours, name
    ours, theirs = (run_scenario(env, "sourcing", {"product": "PX"}) for env in (builtin, neo4j))
    assert theirs["ranking"] == ours["ranking"] and theirs["best"] == {"region_b_share": 0.5}
    for mine, other in zip(ours["branches"], theirs["branches"]):
        assert (other["status"], other["figures"], other["rejected_by"]) == \
            (mine["status"], mine["figures"], mine["rejected_by"])
