"""Tests for the page: sign-in, the CSRF, Origin and Host checks, the serve command and the Configuration page."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
import pytest
from starlette.testclient import TestClient

import clm_portal
from clm_portal import web
from conftest import KEY, ORIGIN, SERVER_KEY, FakeServer, csrf_of, post, sign_in

PAGES = ["/", "/ask", "/review", "/documents", "/jobs/current", "/passage?collection=handbook&id=hb-401k"]


def test_every_page_needs_sign_in(client):
    for path in PAGES:
        response = client.get(path, follow_redirects=False)
        assert response.status_code == 303 and response.headers["location"] == "/sign-in", path
    for path in ["/sign-out", "/ask", "/review/facts/k401_2026.match_cap/approve", "/documents/load"]:
        assert client.post(path, data={"csrf": "x"}, headers={"Origin": ORIGIN}).status_code == 403, path
    assert "Sign in" in client.get("/sign-in").text


def test_a_wrong_key_is_refused(client):
    response = client.post("/sign-in", data={"key": "wrong", "name": "Kevin"}, headers={"Origin": ORIGIN})
    assert response.status_code == 401 and "wrong key" in response.text.lower()
    assert "set-cookie" not in response.headers
    assert client.get("/", follow_redirects=False).status_code == 303
    nameless = client.post("/sign-in", data={"key": KEY, "name": " "}, headers={"Origin": ORIGIN})
    assert nameless.status_code == 400 and client.get("/", follow_redirects=False).status_code == 303


def test_a_signed_in_session_has_a_strict_http_only_cookie_and_can_sign_out(client):
    response = client.post("/sign-in", data={"key": KEY, "name": "Kevin"}, headers={"Origin": ORIGIN})
    cookie = response.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie and "path=/" in cookie
    token = csrf_of(response.text)
    assert "Kevin" in client.get("/").text
    post(client, "/sign-out", token)
    assert client.get("/", follow_redirects=False).status_code == 303


def test_a_post_without_the_csrf_token_is_refused(client):
    token = sign_in(client)
    assert client.post("/sign-out", headers={"Origin": ORIGIN}).status_code == 403
    assert client.post("/sign-out", data={"csrf": "not-it"}, headers={"Origin": ORIGIN}).status_code == 403
    assert client.get("/", follow_redirects=False).status_code == 200
    assert post(client, "/sign-out", token).status_code == 200


def test_a_post_from_another_origin_is_refused(client):
    for headers in ({"Origin": "http://evil.example"}, {}, {"Origin": "null"}):
        assert client.post("/sign-in", data={"key": KEY, "name": "Kevin"}, headers=headers).status_code == 403
    token = sign_in(client)
    assert client.post("/sign-out", data={"csrf": token}, headers={"Origin": "http://evil.example"}).status_code == 403
    assert client.get("/", follow_redirects=False).status_code == 200


def test_a_foreign_host_header_is_refused(client):
    sign_in(client)
    assert client.get("/", headers={"Host": "evil.example:8400"}).status_code == 400
    assert client.get("/sign-in", headers={"Host": "127.0.0.1:8401"}).status_code == 400
    assert client.get("/", headers={"Host": "localhost:8400"}).status_code == 200


def serve_args(environment, *more):
    return clm_portal.parse(["serve", str(environment), *more])


def test_serve_refuses_to_start_without_a_key(environment, monkeypatch, capsys):
    monkeypatch.delenv("CLM_PORTAL_KEY", raising=False)
    ran = []
    assert clm_portal.serve(serve_args(environment), run=lambda *a, **k: ran.append(a)) == 2
    assert "CLM_PORTAL_KEY" in capsys.readouterr().err and not ran
    monkeypatch.setenv("CLM_PORTAL_KEY", KEY)
    other = environment.parent / "rules" / "vesting.dmn"
    assert clm_portal.serve(serve_args(other), run=lambda *a, **k: ran.append(a)) == 2
    assert "not an environment" in capsys.readouterr().err and not ran


def test_serve_listens_on_loopback_by_default(environment, monkeypatch, capsys):
    monkeypatch.setenv("CLM_PORTAL_KEY", KEY)
    ran = []
    assert clm_portal.serve(serve_args(environment), run=lambda app, **kwargs: ran.append(kwargs)) == 0
    assert ran[0]["host"] == "127.0.0.1" and ran[0]["port"] == 8400
    assert "warning" not in capsys.readouterr().err.lower()
    assert clm_portal.serve(serve_args(environment, "--host", "0.0.0.0", "--port", "9000"),
                            run=lambda app, **kwargs: ran.append(kwargs)) == 0
    assert ran[1]["host"] == "0.0.0.0" and ran[1]["port"] == 9000
    err = capsys.readouterr().err
    assert "warning" in err.lower() and "plain HTTP" in err and KEY not in err


def test_configuration_shows_every_rule_and_graph_row_and_points_to_the_inspector(client, environment):
    sign_in(client)
    page = client.get("/").text
    for text in ("Tiny handbook", "A made-up handbook, for the portal", "years_of_service", "vesting",
                 "carryover_limit",
                 "The match vests 25% for each full year.", "People Operations (made up)", "2026-01",
                 "&gt;= 4", "vested_percent", "meridian", "k401_2026", "offered_by", "6300", "years_of_service",
                 "Full years of service.", "handbook", "2026-11-01T09:30:00+00:00", "handbook.jsonl"):
        assert text in page, text
    assert "/passage?collection=handbook&amp;id=hb-401k" in page
    assert f"npx -y @modelcontextprotocol/inspector@2.13.0 uv run clm-server {environment}" in page
    assert "Resource templates" not in page and "document_handbook" not in page
    assert "/passage?collection=handbook&amp;id=hb-leave&amp;passage=%282%29" in page


def test_supplied_and_derived_rules_are_labelled_with_origin_and_status(client):
    sign_in(client)
    page = client.get("/").text
    assert 'class="origin supplied"' in page and 'class="origin derived"' in page and "checked" in page
    assert "confidence" in page and "high-ish" in page


def test_a_passage_is_read_through_the_portals_server(client):
    sign_in(client)
    page = client.get("/passage", params={"collection": "handbook", "id": "hb-401k", "passage": "(4)"}).text
    assert "The passage (4) of hb-401k, as stored." in page
    assert client.get("/passage", params={"collection": "nope", "id": "x"}).status_code == 404


def test_a_server_that_failed_to_start_shows_its_error_and_the_files(environment, portal, monkeypatch):
    monkeypatch.setattr(web, "WRONG_KEY_WAIT", 0.01)
    portal.server = FakeServer(error="error: environment.functions: cannot import 'tiny_functions'")
    with TestClient(web.create_app(portal, KEY, "127.0.0.1", 8400), base_url=ORIGIN) as client:
        sign_in(client)
        page = client.get("/").text
        assert "cannot import &#x27;tiny_functions&#x27;" in page and "not running" in page
        assert "carryover_limit" in page and "k401_2026" in page and "&gt;= 4" in page
        passage = client.get("/passage", params={"collection": "handbook", "id": "hb-401k"}).text
        assert "not running" in passage
    assert environment.is_file()


@pytest.mark.parametrize("path", PAGES + ["/sign-in"])
def test_the_keys_never_appear_on_a_page(client, path, monkeypatch):
    monkeypatch.setenv("CLM_PORTAL_KEY", KEY)
    assert KEY not in client.get("/sign-in").text
    assert KEY not in client.post("/sign-in", data={"key": KEY + "x", "name": "K"}, headers={"Origin": ORIGIN}).text
    signed = client.post("/sign-in", data={"key": KEY, "name": "Kevin"}, headers={"Origin": ORIGIN}).text
    assert KEY not in signed and SERVER_KEY not in signed
    page = client.get(path).text
    assert KEY not in page and SERVER_KEY not in page
