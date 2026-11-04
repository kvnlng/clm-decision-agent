"""Tests for what can go wrong around the page: a server that does not answer or did not start in testing mode, a
derived layer with wrongly typed fields, a correction that fails half way, and what the portal installs."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
import json
import tomllib
from pathlib import Path

import pytest

from clm_portal import layer, managed, web
from conftest import post, sign_in
from test_managed import FakePopen, managed_server
from test_review import FACT, GOOD_DMN, layer_of


def test_a_server_that_does_not_answer_shows_so_on_every_page(client, server):
    sign_in(client)
    server.unreachable = True
    for path in ("/", "/passage?collection=handbook&id=hb-401k", "/review", "/documents", "/ask"):
        response = client.get(path)
        assert response.status_code == 200, path
    assert "not answering" in client.get("/").text
    assert "not answering" in client.get("/passage?collection=handbook&id=hb-401k").text


def test_ask_keeps_the_question_when_the_server_does_not_answer(client, server, portal):
    seen: list[dict] = []
    from test_chat import reply, scripted  # pylint: disable=import-outside-toplevel
    model = scripted([reply("Here.")], seen)
    portal.make_model = lambda *args: model
    token = sign_in(client)
    server.unreachable = True
    response = post(client, "/ask", token, {"question": "Still there?", "backend": "openai"})
    assert response.status_code == 200 and "not answering" in response.text
    assert ">Still there?</textarea>" in response.text
    server.unreachable = False
    post(client, "/ask", token, {"question": "Now?", "backend": "openai"})
    assert [item["content"] for item in seen[0]["messages"] if item["role"] == "user"] == ["Now?"]


def mistyped(environment) -> None:
    file = environment.parent / "derived" / "derived.json"
    found = json.loads(file.read_text())
    found["decisions"]["carryover_limit"].update(rule="oops", source="x", check="yes", dmn=5)
    found["facts"][FACT].update(source=["a"], check={"passed": True, "failures": "none"})
    found["conflicts"][FACT]["source"] = 3
    file.write_text(json.dumps(found))


def test_wrongly_typed_derived_fields_are_load_errors_and_the_rest_is_shown(client, environment):
    mistyped(environment)
    token = sign_in(client)
    configuration = client.get("/")
    assert configuration.status_code == 200 and "carryover_limit" in configuration.text
    assert "derived decisions.carryover_limit.rule" in configuration.text and "vesting" in configuration.text
    review = client.get("/review")
    assert review.status_code == 200 and FACT in review.text
    assert "derived facts.k401_2026.match_cap.source" in review.text
    assert post(client, "/review/decisions/carryover_limit/correct", token, {"dmn": GOOD_DMN}).status_code == 400
    found = layer.configuration(environment, [], None, None)
    assert any("carryover_limit.dmn" in error for error in found["errors"])


@pytest.mark.anyio
async def test_a_server_that_fails_in_testing_mode_is_not_called_testing(tmp_path):
    server = managed_server(FakePopen(code=2, stderr="error: unrecognized arguments: --admit-derived\n"), tmp_path)
    await server.start(testing=True)
    assert server.error and not server.testing


def test_the_banner_needs_a_server_running_in_testing_mode(client, server):
    sign_in(client)
    server.testing, server.error = True, "error: it stopped"
    assert web.TESTING_BANNER not in client.get("/").text


def test_the_portal_installs_the_neo4j_driver():
    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    main = next(item for item in project["project"]["dependencies"] if item.startswith("clm-decision-agent"))
    assert {"documents", "agent", "neo4j"} <= {extra.strip() for extra in main[main.index("[") + 1:main.index("]")]
                                                .split(",")}


def test_a_failed_table_replace_leaves_the_entry_proposed_with_no_check(environment, monkeypatch):
    table = environment.parent / "derived" / "rules" / "carryover_limit.dmn"
    before = table.read_text()

    replace = layer.os.replace

    def refuse(source, target):
        if str(target).endswith(".dmn"):
            raise OSError("disk full")
        replace(source, target)
    monkeypatch.setattr(layer.os, "replace", refuse)
    with pytest.raises(OSError):
        layer.correct_decision(environment, "carryover_limit", GOOD_DMN, by="Kevin", note=None)
    entry = layer_of(environment)["decisions"]["carryover_limit"]
    assert entry["status"] == "proposed" and entry["check"] is None and table.read_text() == before
    assert not list(table.parent.glob(".*.partial"))


def test_a_correction_restarts_the_portals_server(client, server, runner):
    token = sign_in(client)
    runner.release.clear()
    try:
        assert post(client, f"/review/facts/{FACT}/correct", token, {"value": "6300"}).status_code == 200
        assert len(server.starts) == 2
    finally:
        runner.release.set()


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_tool_failed_covers_a_server_that_is_not_answering():
    assert issubclass(managed.NotAnswering, managed.ToolFailed)
