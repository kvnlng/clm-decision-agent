"""Tests for environment.py: loading an environment's graph, queries, decisions, calculations and scenarios."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
from dataclasses import replace

import pytest
from conftest import BASE_TOML, ENV_TOML, JUDGE_TOML, limit_dmn
from fake_documents import FakeEmbedder, FakeStore

from clm_decision.environment import _connector, _graph_settings, is_environment, load_environment
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


class Closable:
    """A store or connector that notes whether it was closed."""

    def __init__(self):
        self.closed = 0

    def close(self):
        self.closed += 1


class ClosableStore(FakeStore, Closable):
    """A fake document store that notes whether it was closed."""

    def __init__(self):
        FakeStore.__init__(self)
        Closable.__init__(self)


WITH_DOCUMENTS = ENV_TOML + '\n[documents]\nbackend = "postgres"\npassword_env = "TOY_DOCUMENTS_PASSWORD"\n'


def test_reuse_shares_clm_and_store(write_environment):
    path = write_environment(WITH_DOCUMENTS + JUDGE_TOML)
    old = load_environment(path, store=ClosableStore(), embedder=FakeEmbedder())
    new = load_environment(path, reuse=old)
    assert new.clm is old.clm and new.store is old.store and new.embedder is old.embedder
    assert new.judges["risk"].name == "risk"
    assert new.graph is not old.graph and new.connector is not old.connector       # the built-in graph is read again


def test_explicit_store_wins_over_reuse(write_environment):
    path = write_environment(WITH_DOCUMENTS)
    old = load_environment(path, store=ClosableStore(), embedder=FakeEmbedder())
    own = ClosableStore()
    assert load_environment(path, store=own, reuse=old).store is own


def test_reuse_rebuilds_when_settings_change(write_environment):
    path = write_environment(WITH_DOCUMENTS + JUDGE_TOML)
    old = load_environment(path, store=ClosableStore(), embedder=FakeEmbedder())
    path.write_text(path.read_text(encoding="utf-8").replace('backend = "mlx"', 'backend = "mlx"\nmodel_id = "other"')
                    .replace('backend = "postgres"', 'backend = "postgres"\nhost = "elsewhere"'), encoding="utf-8")
    other_store = ClosableStore()
    new = load_environment(path, store=other_store, embedder=FakeEmbedder(), reuse=old)
    assert new.clm is not old.clm and new.clm.model_id == "other"
    assert new.store is other_store and new.document_settings != old.document_settings


def test_reuse_connects_again_when_documents_change(write_environment, monkeypatch):
    path = write_environment(WITH_DOCUMENTS)
    old = load_environment(path, store=ClosableStore(), embedder=FakeEmbedder())
    path.write_text(path.read_text(encoding="utf-8").replace('backend = "postgres"',
                                                             'backend = "postgres"\ndatabase = "other"'),
                    encoding="utf-8")
    fresh = ClosableStore()
    monkeypatch.setattr("clm_decision.environment.docs.connect", lambda settings: fresh)
    assert load_environment(path, reuse=old).store is fresh


def test_reuse_shares_a_neo4j_connector_when_graph_settings_are_equal(write_environment):
    path = write_environment()
    neo4j = replace(load_environment(path), connector=Closable(), graph_settings=replace(
        _graph_settings({}, path.parent, "neo4j"), uri="bolt://one"))
    same, changed = neo4j.graph_settings, replace(neo4j.graph_settings, uri="bolt://two")
    assert _connector(same, None, neo4j.functions, neo4j) is neo4j.connector
    with pytest.raises(ScenarioError, match="password_env"):        # different settings: a new driver is wanted
        _connector(changed, None, neo4j.functions, neo4j)


def test_close_keeps_shared(write_environment):
    path = write_environment(WITH_DOCUMENTS)
    store = ClosableStore()
    old = load_environment(path, store=store, embedder=FakeEmbedder())
    new = load_environment(path, reuse=old)
    old.close(keeping=new)
    assert store.closed == 0
    old.close()
    assert store.closed == 1


def test_close_closes_a_store_the_new_environment_does_not_hold(write_environment):
    path = write_environment(WITH_DOCUMENTS)
    old_store, new_store = ClosableStore(), ClosableStore()
    old = load_environment(path, store=old_store, embedder=FakeEmbedder())
    new = load_environment(path, store=new_store, embedder=FakeEmbedder())
    old.close(keeping=new)
    assert old_store.closed == 1 and new_store.closed == 0


def test_close_closes_the_neo4j_connector_unless_kept(write_environment):
    path = write_environment()
    connector = Closable()
    old = replace(load_environment(path), connector=connector)
    old.close(keeping=replace(old))
    assert connector.closed == 0
    old.close()
    assert connector.closed == 1


def test_a_store_with_nothing_to_close_is_fine(write_environment):
    load_environment(write_environment()).close()
    load_environment(write_environment(WITH_DOCUMENTS, folder="second"), store=FakeStore(),
                     embedder=FakeEmbedder()).close()


class FakeConnection:
    """A database connection that notes being closed."""
    closed, broken = False, False

    def close(self):
        self.closed = True


def test_a_closed_postgres_store_stays_closed():
    from clm_decision.documents_pg import PostgresStore  # pylint: disable=import-outside-toplevel
    opened = []
    store = PostgresStore(FakeConnection(), "here", reconnect=lambda: opened.append(1) or FakeConnection())
    first = store.connection
    store.close()
    store.close()
    assert first.closed and store._connection() is first and not opened     # pylint: disable=protected-access


def test_a_load_that_fails_closes_the_store_it_connected(write_environment, monkeypatch):
    fresh = ClosableStore()
    monkeypatch.setattr("clm_decision.environment.docs.connect", lambda settings: fresh)

    def fail(*args):
        raise ScenarioError("derived: broken")
    monkeypatch.setattr("clm_decision.environment._fill", fail)
    with pytest.raises(ScenarioError, match="derived: broken"):
        load_environment(write_environment(WITH_DOCUMENTS))
    assert fresh.closed == 1


def test_a_load_that_fails_leaves_a_shared_store_open(write_environment, monkeypatch):
    path = write_environment(WITH_DOCUMENTS)
    old = load_environment(path, store=ClosableStore(), embedder=FakeEmbedder())

    def fail(*args):
        raise ScenarioError("derived: broken")
    monkeypatch.setattr("clm_decision.environment._fill", fail)
    with pytest.raises(ScenarioError):
        load_environment(path, reuse=old)
    assert old.store.closed == 0


def test_a_bad_second_reader_is_refused_before_a_store_is_connected(write_environment, monkeypatch):
    connected = []
    monkeypatch.setattr("clm_decision.environment.docs.connect", lambda settings: connected.append(ClosableStore()))
    with pytest.raises(ScenarioError, match="^derived.second_reader: must be a table"):
        load_environment(write_environment(WITH_DOCUMENTS + '\n[derived]\nsecond_reader = "nope"\n'))
    assert not connected
