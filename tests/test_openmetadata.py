"""Tests for the service desk's OpenMetadata reader and its mock server."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,missing-class-docstring,redefined-outer-name,unused-argument
import pytest
from langchain_core.tools import ToolException
from openmetadata_catalogue import fetch, to_entries, upstream


def table(name, description="A table", owners=(), tags=(), domains=()):
    return {"id": f"id-{name}", "name": name, "fullyQualifiedName": f"db.schema.{name}", "description": description,
            "owners": [{"name": owner, "type": "team"} for owner in owners],
            "tags": [{"tagFQN": tag} for tag in tags], "domains": [{"name": domain} for domain in domains]}


def lineage(name, *sources):
    return {"entity": {"id": f"id-{name}"},
            "nodes": [{"id": f"id-{source}", "name": source} for source in sources],
            "upstreamEdges": [{"fromEntity": f"id-{source}", "toEntity": f"id-{name}"} for source in sources],
            "downstreamEdges": []}


def test_facts_come_from_owners_tags_and_lineage():
    tables = [table("Readmissions Extract", owners=["Quality Improvement"], tags=["PII.NonSensitive", "Tier.Tier2"]),
              table("ED Attendances", tags=["Custom.Thing"])]
    built_from = {"db.schema.Readmissions Extract": lineage("Readmissions Extract", "ED Attendances")}
    entries = to_entries(tables, built_from)
    first, second = entries[0], entries[1]
    assert first == {"name": "Readmissions Extract", "description": "A table", "facts": [
        "Quality Improvement owns Readmissions Extract.",
        "Readmissions Extract contains personal data that is not sensitive.",
        "Readmissions Extract is a tier 2 dataset.",
        "Readmissions Extract is built from ED Attendances."]}
    assert second["facts"] == ["ED Attendances is tagged Custom.Thing."]


def test_a_sensitive_tag_and_an_owner_s_display_name():
    owners = [{"name": "cancer-services", "displayName": "Cancer Services", "type": "team"}]
    entry = to_entries([dict(table("Oncology Registry", tags=["PII.Sensitive"]), owners=owners)], {})[0]
    assert entry["facts"] == ["Cancer Services owns Oncology Registry.",
                              "Oncology Registry contains sensitive personal data."]


def test_markdown_is_stripped_and_undescribed_tables_are_left_out():
    entries = to_entries([table("A", description="**Cancer** data, see [the wiki](http://x).\n\n# Notes"),
                          table("B", description=""), table("C", description=None)], {})
    assert entries == [{"name": "A", "description": "Cancer data, see the wiki. Notes", "facts": []}]


def test_missing_lists_are_treated_as_empty():
    assert to_entries([{"name": "A", "description": "Admissions"}], {"db.schema.A": {}}) == [
        {"name": "A", "description": "Admissions", "facts": []}]


def test_fetch_pages_through_the_tables_and_reads_their_lineage(openmetadata_server, openmetadata_url):
    tables, lineage = fetch(openmetadata_url, None, page=3)
    assert len(tables) == 8
    assert "Oncology Registry" in [table["name"] for table in tables]
    assert upstream(lineage["warehouse.clinical.readmissions_extract"]) == ["Emergency Department Attendances"]
    assert upstream(lineage["warehouse.clinical.oncology_registry"]) == []
    assert sum(path.startswith("/api/v1/tables") for path in openmetadata_server.requests) == 3


def test_fetch_keeps_only_the_named_domain(openmetadata_url):
    tables, lineage = fetch(openmetadata_url, None, domain="Workforce")
    assert [table["name"] for table in tables] == ["Staffing Rosters"]
    assert list(lineage) == ["warehouse.clinical.staffing_rosters"]


def test_fetch_sends_the_token(openmetadata_server, openmetadata_url):
    fetch(openmetadata_url, "secret")
    assert openmetadata_server.last_authorization == "Bearer secret"


def test_fetch_reports_an_error_status_with_the_address(openmetadata_url):
    with pytest.raises(ToolException, match=r"OpenMetadata answered 404 for http://.*/wrong/api/v1/tables"):
        fetch(openmetadata_url + "/wrong", None)


def test_fetch_reports_a_server_that_cannot_be_reached():
    with pytest.raises(ToolException, match=r"OpenMetadata at http://127\.0\.0\.1:9 could not be reached"):
        fetch("http://127.0.0.1:9", None)


def test_the_mock_s_tables_become_eight_entries_with_facts(openmetadata_url):
    entries = {entry["name"]: entry for entry in to_entries(*fetch(openmetadata_url, None))}
    assert len(entries) == 8
    assert entries["Oncology Registry"]["facts"] == ["Cancer Services owns Oncology Registry.",
                                                     "Oncology Registry contains sensitive personal data."]
    assert "Readmissions Extract is built from Emergency Department Attendances." in \
        entries["Readmissions Extract"]["facts"]


def test_a_table_name_used_in_two_schemas_is_given_its_full_name():
    first = dict(table("admissions", owners=["Finance"]), fullyQualifiedName="warehouse.finance.admissions")
    second = dict(table("admissions"), fullyQualifiedName="warehouse.clinical.admissions")
    names = [entry["name"] for entry in to_entries([first, second, table("ward_list")], {})]
    assert names == ["warehouse.finance.admissions", "warehouse.clinical.admissions", "ward_list"]
    assert to_entries([first, second], {})[0]["facts"] == ["Finance owns warehouse.finance.admissions."]
