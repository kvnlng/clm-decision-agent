"""Tests for what can go wrong around the page: a reload that fails, a derived layer with wrongly typed fields, a
correction that fails half way, and what the portal installs."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
import asyncio
import json
import tomllib
from pathlib import Path

import pytest

from clm_decision.portal import layer, live, web
from .conftest import post, sign_in
from .test_review import FACT, GOOD_DMN, layer_of


def test_a_failed_reload_is_shown_on_configuration_and_every_page_still_serves(client, server):
    token = sign_in(client)
    server.failing = "the last reload failed: environment.functions: cannot import 'x'"
    response = post(client, "/reload", token)
    assert response.status_code == 409 and "cannot import &#x27;x&#x27;" in response.text
    for path in ("/", "/passage?collection=handbook&id=hb-401k", "/review", "/documents", "/ask"):
        assert client.get(path).status_code == 200, path
    assert "The passage (1) of hb-401k, as stored." in client.get("/passage?collection=handbook&id=hb-401k").text
    server.failing = None
    portal = client.app.app.state.portal
    asyncio.run(portal.jobs.lock.acquire())     # as while a job runs
    busy = post(client, "/reload", token)
    portal.jobs.lock.release()
    assert busy.status_code == 409 and "wait for it to finish" in busy.text
    assert post(client, "/reload", token, follow_redirects=False).status_code == 303
    assert "cannot import" not in client.get("/").text and server.reloads == [False, False]


def test_ask_still_answers_after_a_failed_reload(client, server, portal):
    seen: list[dict] = []
    from .test_chat import reply, scripted  # pylint: disable=import-outside-toplevel
    portal.make_model = lambda *args: scripted([reply("Here.")], seen)
    token = sign_in(client)
    server.failing = "the last reload failed: a broken file"
    post(client, "/reload", token)
    response = post(client, "/ask", token, {"question": "Still there?", "backend": "openai"})
    assert response.status_code == 200 and "Here." in response.text
    assert "decision_vesting" in [tool["function"]["name"] for tool in seen[0]["tools"]]


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
    found = layer.configuration(environment, None, None)
    assert any("carryover_limit.dmn" in error for error in found["errors"])


def test_the_banner_shows_only_while_testing(client, server):
    sign_in(client)
    server.error = "the last reload failed: it stopped"
    assert web.TESTING_BANNER not in client.get("/").text
    server.testing = True
    assert web.TESTING_BANNER in client.get("/").text


def test_the_dev_group_installs_the_extras_the_portal_needs():
    project = tomllib.loads((Path(__file__).parents[2] / "pyproject.toml").read_text())
    main = next(item for item in project["dependency-groups"]["dev"] if item.startswith("clm-decision-agent"))
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


def test_a_correction_reloads_the_server(client, server, runner):
    token = sign_in(client)
    runner.release.clear()
    try:
        assert post(client, f"/review/facts/{FACT}/correct", token, {"value": "6300"}).status_code == 200
        assert server.reloads == [False]
    finally:
        runner.release.set()


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_tool_failed_covers_a_server_that_is_not_answering():
    assert issubclass(live.NotAnswering, live.ToolFailed)
