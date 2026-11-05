"""Tests for Review: derived entries beside their passages and checks, and every change to their status."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
import hashlib
import json
import re

import pytest
from starlette.testclient import TestClient

from clm_portal import layer, web
from conftest import KEY, ORIGIN, FakeServer, dmn, post, sign_in, source

FACT = "k401_2026.match_cap"
GOOD_DMN = dmn("carryover_limit", "limit", [("&gt;= 5", "10"), ("-", "5")])


def layer_of(environment) -> dict:
    return json.loads((environment.parent / "derived" / "derived.json").read_text())


def hashes(folder, skip_derived=True) -> dict:
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(folder.rglob("*"))
            if path.is_file() and not (skip_derived and "derived" in path.relative_to(folder).parts)}


def test_review_shows_each_entry_beside_its_passage_and_check(client):
    sign_in(client)
    page = client.get("/review").text
    assert page.index("carryover_limit") < page.index(FACT)
    assert "check passed" in page and "check failed" in page and "the quote does not state 5000" in page
    assert "The passage (2) of hb-leave, as stored." in page and "The passage (4) of hb-401k, as stored." in page
    assert "&gt;= 5" in page and "Up to 7 unused days carry over." in page


def test_without_the_server_the_quote_is_shown(portal, monkeypatch):
    monkeypatch.setattr(web, "WRONG_KEY_WAIT", 0.01)
    portal.server = FakeServer(error="error: cannot reach PostgreSQL")
    with TestClient(web.create_app(portal, KEY, "127.0.0.1", 8400), base_url=ORIGIN) as client:
        sign_in(client)
        page = client.get("/review").text
    assert "up to 7 unused days carry into the next year" in page and "as stored" not in page


def test_approve_records_the_signed_in_name(client, environment, server):
    token = sign_in(client, "Kevin")
    response = post(client, "/review/decisions/carryover_limit/approve", token, {"note": "matches the handbook"})
    assert response.status_code == 200 and "approved" in response.text
    entry = layer_of(environment)["decisions"]["carryover_limit"]
    assert entry["status"] == "approved" and entry["decided_by"] == "Kevin" and entry["note"] == "matches the handbook"
    assert entry["decided_at"] and len(server.starts) == 2


def test_a_failed_check_cannot_be_approved(client, environment):
    token = sign_in(client)
    page = client.get("/review").text
    fact_part = page[page.index(f'id="facts-{FACT}"'):]
    assert "disabled" in fact_part[:fact_part.index("</section>")].split("Approve")[0][-200:]
    response = post(client, f"/review/facts/{FACT}/approve", token)
    assert response.status_code == 409 and "cannot be approved" in response.text
    assert layer_of(environment)["facts"][FACT]["status"] == "proposed"
    assert post(client, f"/review/facts/{FACT}/reject", token).status_code == 200
    assert layer_of(environment)["facts"][FACT]["status"] == "rejected"


def test_a_conflict_shows_the_supplied_value_and_can_be_resolved(client, environment):
    token = sign_in(client)
    page = client.get("/review").text
    conflict = page[page.index(f'id="conflicts-{FACT}"'):]
    conflict = conflict[:conflict.index("</section>")]
    assert "6300" in conflict and "5000" in conflict and "401(k) plan" in conflict
    assert post(client, f"/review/conflicts/{FACT}/resolve", token, {"note": "the plan is right"}).status_code == 200
    entry = layer_of(environment)["conflicts"][FACT]
    assert entry["status"] == "resolved" and entry["decided_by"] == "Kevin"
    assert post(client, f"/review/conflicts/{FACT}/approve", token).status_code == 400


def test_a_correction_clears_the_check_sets_proposed_and_runs_clm_derive_check(client, environment, runner):
    token = sign_in(client)
    response = post(client, f"/review/facts/{FACT}/correct", token, {"value": "6300", "note": "as the plan says"})
    assert response.status_code == 200
    entry = layer_of(environment)["facts"][FACT]
    assert entry["value"] == 6300 and entry["check"] is None and entry["status"] == "proposed"
    assert entry["decided_by"] == "Kevin" and entry["note"] == "as the plan says"
    response = post(client, "/review/decisions/carryover_limit/correct", token, {"dmn": GOOD_DMN})
    assert response.status_code == 200
    entry = layer_of(environment)["decisions"]["carryover_limit"]
    assert entry["check"] is None and entry["status"] == "proposed"
    assert (environment.parent / "derived" / "rules" / "carryover_limit.dmn").read_text() == GOOD_DMN
    assert [argv[1:] for argv in runner.seen] == [["check", str(environment)]] * 2
    assert runner.seen[0][0].endswith("clm-derive")
    assert post(client, "/review/decisions/carryover_limit/approve", token).status_code == 409
    assert post(client, f"/review/facts/{FACT}/correct", token, {"value": "{not json"}).status_code == 400


def test_a_dmn_outside_derived_is_refused(client, environment):
    file = environment.parent / "derived" / "derived.json"
    file.write_text(file.read_text().replace('"rules/carryover_limit.dmn"', '"../rules/vesting.dmn"'))
    before = hashes(environment.parent, skip_derived=False)
    token = sign_in(client)
    response = post(client, "/review/decisions/carryover_limit/correct", token, {"dmn": GOOD_DMN})
    assert response.status_code == 400 and "outside derived/" in response.text
    assert hashes(environment.parent, skip_derived=False) == before


def test_an_unreadable_dmn_is_refused_and_nothing_is_written(client, environment, runner):
    before = hashes(environment.parent, skip_derived=False)
    token = sign_in(client)
    for text in ("<not a table", dmn("carryover_limit", "limit", [("between 1 and 2", "7")])):
        response = post(client, "/review/decisions/carryover_limit/correct", token, {"dmn": text})
        assert response.status_code == 400 and "carryover_limit" in response.text
    assert hashes(environment.parent, skip_derived=False) == before and not runner.seen


def test_a_change_while_a_job_runs_is_refused(client, environment, runner):
    token = sign_in(client)
    runner.release.clear()
    try:
        assert post(client, f"/review/facts/{FACT}/correct", token, {"value": "6300"}).status_code == 200
        response = post(client, "/review/decisions/carryover_limit/approve", token)
        assert response.status_code == 409 and "wait" in response.text
        assert layer_of(environment)["decisions"]["carryover_limit"]["status"] == "checked"
    finally:
        runner.release.set()


def test_no_supplied_file_changes(client, environment):
    before = hashes(environment.parent)
    token = sign_in(client)
    post(client, "/review/decisions/carryover_limit/approve", token)
    post(client, f"/review/conflicts/{FACT}/resolve", token)
    post(client, f"/review/facts/{FACT}/correct", token, {"value": "6300"})
    post(client, "/review/decisions/carryover_limit/correct", token, {"dmn": GOOD_DMN})
    post(client, f"/review/facts/{FACT}/reject", token)
    assert hashes(environment.parent) == before
    assert layer.review(environment)["groups"]


def test_review_groups_checked_first_then_proposed_then_decided(environment):
    groups = layer.review(environment)["groups"]
    assert [title for title, _ in groups][:3] == ["Checked", "Open conflicts", "Proposed"]
    fact = next(item for title, items in groups if title == "Proposed" for item in items)
    assert fact["supplied"] == "6300" and fact["graph_row"]["name"] == "401(k) plan"


def test_a_rejected_entry_that_passed_its_check_can_be_approved_again(client, environment):
    token = sign_in(client, "Kevin")
    assert post(client, "/review/decisions/carryover_limit/reject", token).status_code == 200
    page = client.get("/review").text
    part = page[page.index('id="decisions-carryover_limit"'):]
    assert "disabled" not in part[:part.index("</section>")].split("Approve")[0][-200:]
    assert post(client, "/review/decisions/carryover_limit/approve", token).status_code == 200
    assert layer_of(environment)["decisions"]["carryover_limit"]["status"] == "approved"


def test_an_entry_and_a_document_whose_names_hold_a_slash_can_be_acted_on(client, environment):
    file = environment.parent / "derived" / "derived.json"
    found = json.loads(file.read_text())
    name = "dept/sales.cap"
    found["facts"][name] = {**found["facts"][FACT], "node": "dept/sales", "property": "cap",
                            "source": source("hb/leave", "(2)", "up to 7"), "status": "checked",
                            "check": {"passed": True, "failures": []}}
    file.write_text(json.dumps(found))
    token = sign_in(client)
    page = client.get("/review").text
    assert "/review/facts/dept%2Fsales.cap/approve" in page
    assert "/passage?collection=handbook&amp;id=hb%2Fleave&amp;passage=%282%29" in page
    assert "The passage (2) of hb/leave, as stored." in page
    assert post(client, "/review/facts/dept%2Fsales.cap/approve", token).status_code == 200
    assert layer_of(environment)["facts"][name]["status"] == "approved"
    assert post(client, "/review/facts/dept%2Fsales.cap/correct", token, {"value": "1"}).status_code == 200
    assert layer_of(environment)["facts"][name]["value"] == 1
    assert "The passage (1) of hb/leave" in client.get("/passage", params={"collection": "handbook",
                                                                           "id": "hb/leave"}).text


@pytest.mark.parametrize("installed", [{"clm-server", "clm-load-documents"}])
def test_a_correction_without_clm_derive_says_so_as_an_error(client, environment):
    token = sign_in(client)
    response = post(client, f"/review/facts/{FACT}/correct", token, {"value": "6300"})
    assert response.status_code == 200 and layer_of(environment)["facts"][FACT]["value"] == 6300
    shown = re.search(r'<p class="(\w+)">[^<]*clm-derive is not installed', response.text)
    assert shown and shown.group(1) == "failed"


def test_with_a_second_reader_an_entry_is_approved_only_after_its_reading(client, environment):
    with environment.open("a") as file:
        file.write('\n[derived]\nsecond_reader = { backend = "ollama", model = "qwen3:14b" }\n')
    token = sign_in(client, "Kevin")
    response = post(client, "/review/decisions/carryover_limit/approve", token)
    assert response.status_code == 409 and "second reading by qwen3:14b" in response.text
    assert layer_of(environment)["decisions"]["carryover_limit"]["status"] == "checked"


def test_a_conflict_over_a_node_whose_id_holds_a_dot_shows_that_nodes_row(environment):
    nodes = environment.parent / "graph" / "nodes.csv"
    nodes.write_text(nodes.read_text() + "site.a,site,Site A,\nsite,site,Not this one,\n")
    file = environment.parent / "derived" / "derived.json"
    found = json.loads(file.read_text())
    found["facts"]["site.a.rent"] = {**found["facts"][FACT], "node": "site.a", "property": "rent"}
    found["conflicts"]["site.a.rent"] = {**found["conflicts"][FACT], "name": "site.a.rent"}
    file.write_text(json.dumps(found))
    items = [item for _, group in layer.review(environment)["groups"] for item in group]
    conflict = next(item for item in items if item["kind"] == "conflicts" and item["name"] == "site.a.rent")
    assert conflict["graph_row"]["name"] == "Site A"


def test_a_correction_is_one_write_that_records_who_made_it(environment, monkeypatch):
    writes = []
    write = layer.derived.write
    monkeypatch.setattr(layer.derived, "write", lambda *args: (writes.append(1), write(*args)))
    layer.correct_fact(environment, FACT, 6000, by="Kevin", note="the cap is 6000")
    entry = layer_of(environment)["facts"][FACT]
    assert len(writes) == 1 and entry["value"] == 6000 and entry["status"] == "proposed" and entry["check"] is None
    assert entry["decided_by"] == "Kevin" and entry["decided_at"] and entry["note"] == "the cap is 6000"


def test_a_correction_with_a_bad_second_reader_table_still_needs_no_second_reader(environment):
    with environment.open("a") as file:
        file.write('\n[derived]\nsecond_reader = "oops"\n')
    layer.correct_decision(environment, "carryover_limit", GOOD_DMN, by="Kevin", note=None)
    entry = layer_of(environment)["decisions"]["carryover_limit"]
    assert entry["status"] == "proposed" and entry["check"] is None and entry["decided_by"] == "Kevin"
    assert "10" in (environment.parent / "derived" / "rules" / "carryover_limit.dmn").read_text()
    with pytest.raises(layer.Refused):
        layer.decide(environment, "decisions", "carryover_limit", "approved", by="Kevin", note=None)


@pytest.mark.parametrize("text", ["[]", "null", '"x"', "3"])
def test_a_derived_layer_that_is_not_an_object_is_a_load_error(environment, text):
    (environment.parent / "derived" / "derived.json").write_text(text)
    found = layer.configuration(environment, None, None)
    assert any("not a derived layer of version 1" in error for error in found["errors"])
    assert any("not a derived layer" in error for error in layer.review(environment)["errors"])
    with pytest.raises(layer.Refused):
        layer.correct_fact(environment, FACT, 1, by="Kevin", note=None)
