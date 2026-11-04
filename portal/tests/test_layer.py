"""Tests for the configuration as the portal reads it from the files, without loading the environment."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name,unsubscriptable-object
from types import SimpleNamespace

from clm_portal import layer

TOOLS = [SimpleNamespace(name="describe", description="What it offers.", input_schema={"properties": {}}),
         SimpleNamespace(name="decision_vesting", description="Vesting.",
                         input_schema={"properties": {"years_of_service": {"type": "number"}}})]
DESCRIBED = {"collections": {"handbook": {"built": {"built_at": "2026-11-01T09:30:00+00:00"}}}}


def by_name(rules):
    return {rule["name"]: rule for rule in rules}


def test_supplied_and_derived_rules_carry_origin_status_and_rows(environment):
    found = layer.configuration(environment, TOOLS, DESCRIBED, None)
    rules = by_name(found["rules"])
    assert rules["vesting"]["origin"] == "supplied" and rules["vesting"]["status"] is None
    assert rules["vesting"]["rule"]["owner"] == "People Operations (made up)"
    assert [row["when"] for row in rules["vesting"]["table"]["rows"]] == [[">= 4"], ["-"]]
    derived = rules["carryover_limit"]
    assert derived["origin"] == "derived" and derived["status"] == "checked" and derived["check"]["passed"]
    assert derived["table"]["outputs"] == ["limit"] and derived["other"] == {"confidence": "high-ish"}
    assert not found["errors"]


def test_tools_information_graph_and_collections_come_from_the_files_and_the_server(environment):
    found = layer.configuration(environment, TOOLS, DESCRIBED, None)
    assert found["name"] == "Tiny handbook"
    assert [tool["parameters"] for tool in found["tools"]] == [[], ["years_of_service"]]
    assert found["information"][0]["name"] == "years_of_service"
    assert found["graph"]["nodes"]["columns"] == ["id", "kind", "name", "match_cap"]
    assert [row["id"] for row in found["graph"]["nodes"]["rows"]] == ["meridian", "k401_2026"]
    assert found["graph"]["edges"]["rows"] == [{"source": "k401_2026", "relation": "offered_by", "target": "meridian"}]
    assert found["collections"][0]["built"]["built_at"] == "2026-11-01T09:30:00+00:00"


def test_a_graph_in_neo4j_is_named_not_shown(environment):
    text = environment.read_text().replace('backend = "builtin"\nnodes = "graph/nodes.csv"\nedges = "graph/edges.csv"',
                                           'backend = "neo4j"')
    environment.write_text(text)
    found = layer.configuration(environment, [], None, None)
    assert found["graph"]["note"] == "graph in Neo4j, not shown" and found["graph"]["nodes"] is None


def test_every_problem_is_a_load_error_and_the_rest_is_still_shown(environment):
    (environment.parent / "rules" / "vesting.dmn").write_text("<not a table")
    (environment.parent / "graph" / "edges.csv").unlink()
    found = layer.configuration(environment, [], None, "the server exited")
    errors = " ".join(found["errors"])
    assert "the server exited" in errors and "vesting.dmn" in errors and "edges.csv" in errors
    assert "carryover_limit" in by_name(found["rules"]) and found["graph"]["nodes"]["rows"]


def test_a_derived_dmn_outside_derived_is_not_read(environment):
    text = (environment.parent / "derived" / "derived.json").read_text()
    (environment.parent / "derived" / "derived.json").write_text(
        text.replace('"rules/carryover_limit.dmn"', '"../rules/vesting.dmn"'))
    found = layer.configuration(environment, [], None, None)
    assert by_name(found["rules"])["carryover_limit"]["table"] is None
    assert any("outside derived/" in error for error in found["errors"])
