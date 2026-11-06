"""Tests for checks.py: the server's short list of checks, each pass, warn, fail or not configured."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
import dataclasses
import json
import os

import pytest
from conftest import ENV_TOML, JUDGE_TOML
from fake_documents import FakeEmbedder
from test_environment_resources import copied, load as load_handbook

from clm_decision import derived, tracing
from clm_decision.admin import Admin
from clm_decision.checks import Checks
from clm_decision.environment import load_environment
from clm_decision.environment_server import build_environment_server, infrastructure, reload
from clm_decision.scenario import ScenarioError


@pytest.fixture
def anyio_backend():
    return "asyncio"


def by_name(checks) -> dict:
    return {check.name: check for check in checks}


@pytest.mark.anyio
async def test_the_checks_come_in_order_and_those_without_their_part_are_not_configured(write_environment):
    found = await Checks(build_environment_server(load_environment(write_environment())), None, None).run()
    assert [check.name for check in found] == ["clm", "postgres", "neo4j", "oauth", "trace", "tools", "describe"]
    statuses = {check.name: check.status for check in found}
    assert statuses == {"clm": "not configured", "postgres": "not configured", "neo4j": "not configured",
                        "oauth": "not configured", "trace": "not configured", "tools": "pass", "describe": "pass"}


@pytest.mark.anyio
async def test_clm_warn_before_load_not_configured_without(write_environment, fake_clm):
    server = build_environment_server(load_environment(write_environment(ENV_TOML + JUDGE_TOML)))
    checks = Checks(server, None, None)
    before = by_name(await checks.run())["clm"]
    with server.holder.use() as env:
        env.clm.probability(env.judges["risk"], "a strike at the port")
    after = by_name(await checks.run())["clm"]
    assert (before.status, before.detail, before.tools) == ("warn", "loads on first judgement", ["judge_risk"])
    assert after.status == "pass" and "mlx" in after.detail
    assert fake_clm.loads == 1


@pytest.mark.anyio
async def test_postgres_fail_names_tools():
    env = load_handbook()
    checks = Checks(build_environment_server(env), None, None)
    up = by_name(await checks.run())["postgres"]
    env.store.down = "connection refused"
    down = by_name(await checks.run())["postgres"]
    assert up.status == "pass" and up.last_error is None
    assert down.status == "fail" and "connection refused" in down.detail
    assert "document_handbook" in down.tools and "search_handbook" in down.tools
    assert down.last_success == up.measured_at and down.last_error == down.detail


@pytest.mark.anyio
async def test_neo4j_runs_return_1_through_a_connector_that_is_not_the_built_in_one(write_environment):
    class Down:  # pylint: disable=too-few-public-methods
        """A Neo4j connector whose database does not answer."""
        def ping(self):
            raise RuntimeError("Neo4j at bolt://127.0.0.1:7687: unreachable\nmore")
    env = load_environment(write_environment())
    found = by_name(await Checks(build_environment_server(dataclasses.replace(env, connector=Down())), None,
                                 None).run())["neo4j"]
    assert found.status == "fail" and found.detail == "RuntimeError: Neo4j at bolt://127.0.0.1:7687: unreachable"
    assert found.tools == ["query_suppliers"]


@pytest.mark.anyio
async def test_fingerprint_ignores_disable(write_environment, tmp_path):
    admin = Admin(tmp_path / "disabled_tools.json")
    server = build_environment_server(load_environment(write_environment()), admin=admin)
    checks = Checks(server, None, None)
    admin.disable("query_suppliers")
    assert by_name(await checks.run())["tools"].status == "pass"


@pytest.mark.anyio
async def test_fingerprint_rebaselines_on_reload(tmp_path):
    server = build_environment_server(copied(tmp_path, {"tuition_eligible": "proposed"}))
    checks = Checks(server, None, None)
    folder = tmp_path / "handbook"
    layer = derived.read(folder)
    layer["decisions"]["tuition_eligible"]["status"] = "approved"
    derived.write(folder, layer)
    before = checks.baseline
    changed = reload(server, load_environment(folder / "environment_derived.toml", store=server.holder.current.store,
                                              embedder=FakeEmbedder()))
    found = by_name(await checks.run())["tools"]
    assert changed["added"] == ["decision_tuition_eligible"]
    assert found.status == "pass" and checks.baseline != before


@pytest.mark.anyio
async def test_fingerprint_fails_on_vanished_tool(write_environment):
    server = build_environment_server(load_environment(write_environment()))
    checks = Checks(server, None, None)
    server.remove_tool("query_suppliers")
    found = by_name(await checks.run())["tools"]
    assert found.status == "fail" and "query_suppliers" in found.detail and "vanished" in found.detail


@pytest.mark.anyio
async def test_trace_check_reads_stopped(write_environment, tmp_path):
    writer = tracing.TraceWriter(tmp_path / "trace.jsonl")
    checks = Checks(build_environment_server(load_environment(write_environment()), trace=writer), writer, None)
    fine = by_name(await checks.run())["trace"]
    writer.close()
    os.chmod(writer.path, 0o644)
    loose = by_name(await checks.run())["trace"]
    writer._stopped = True  # pylint: disable=protected-access
    stopped = by_name(await checks.run())["trace"]
    assert fine.status == "pass"
    assert loose.status == "warn" and "0644" in loose.detail
    assert stopped.status == "fail"


@pytest.mark.anyio
async def test_synthetic_describe_marked(write_environment, tmp_path):
    writer = tracing.TraceWriter(tmp_path / "trace.jsonl")
    checks = Checks(build_environment_server(load_environment(write_environment()), trace=writer), None, None)
    found = by_name(await checks.run())["describe"]
    writer.close()
    records = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert found.status == "pass"
    assert [(record["tool"], record.get("synthetic")) for record in records] == [("describe", True)]
    assert tracing.SYNTHETIC.get() is False


def test_infrastructure_is_declared_and_filled_in(write_environment):
    toml = ENV_TOML.replace('function = "cost"\n', 'function = "cost"\ninfrastructure = ["postgres"]\n')
    env = load_environment(write_environment(toml + JUDGE_TOML))
    found = infrastructure(env)
    assert found["calculation_cost"] == {"postgres"} and found["judge_risk"] == {"clm"}
    assert found["query_suppliers"] == set() and found["describe"] == set()
    assert infrastructure(load_handbook())["document_handbook"] == {"postgres"}


@pytest.mark.parametrize("value", ['"postgres"', '["redis"]', "[1]"])
def test_an_infrastructure_outside_the_three_is_refused_naming_the_table(write_environment, value):
    toml = ENV_TOML.replace('function = "cost"\n', f'function = "cost"\ninfrastructure = {value}\n')
    with pytest.raises(ScenarioError, match=r"calculations\.cost\.infrastructure"):
        load_environment(write_environment(toml))
