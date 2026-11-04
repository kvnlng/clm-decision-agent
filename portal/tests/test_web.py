"""Tests for the page: sign-in, the CSRF, Origin and Host checks, the serve command and the Configuration page."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
import html
import json
from pathlib import Path

import pytest
from mcp.server import MCPServer
from starlette.testclient import TestClient

import clm_portal
from clm_portal import managed, web
from conftest import KEY, ORIGIN, FakeServer, csrf_of, post, sign_in
from test_managed import FakePopen

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


PUBLIC = "https://portal.example"


def test_a_public_origin_behind_a_tls_proxy_is_allowed_only_when_named(portal, monkeypatch):
    monkeypatch.setattr(web, "WRONG_KEY_WAIT", 0.01)
    proxied = {"Origin": PUBLIC, "Host": "portal.example"}
    rewritten = {"Origin": PUBLIC, "Host": "127.0.0.1:8400"}
    form = {"key": KEY, "name": "Kevin"}
    with TestClient(web.create_app(portal, KEY, "127.0.0.1", 8400), base_url=ORIGIN) as client:
        assert client.post("/sign-in", data=form, headers=proxied).status_code == 400
        assert client.post("/sign-in", data=form, headers=rewritten).status_code == 403
    with TestClient(web.create_app(portal, KEY, "127.0.0.1", 8400, public=PUBLIC),
                    base_url=ORIGIN) as client:
        assert client.post("/sign-in", data=form, headers=proxied).status_code == 200
        assert client.post("/sign-in", data=form, headers=rewritten).status_code == 200
        assert post(client, "/sign-out", sign_in(client)).status_code == 200
        assert client.post("/sign-in", data=form, headers={"Origin": "https://evil.example",
                                                          "Host": "portal.example"}).status_code == 403
        assert client.get("/sign-in", headers={"Host": "evil.example"}).status_code == 400


@pytest.mark.parametrize("given, kept", [("https://portal.example/", PUBLIC), ("HTTPS://Portal.Example:443", PUBLIC),
                                         ("https://portal.example:8443", "https://portal.example:8443"),
                                         ("http://10.0.0.5:8080", "http://10.0.0.5:8080")])
def test_a_public_origin_is_kept_as_a_browser_sends_it(given, kept):
    assert web.public_origin(given) == kept


@pytest.mark.parametrize("given", ["portal.example", "https://portal.example/portal", "ftp://portal.example",
                                   "https://portal.example?x=1", "https://"])
def test_serve_refuses_a_public_origin_that_is_not_one(environment, monkeypatch, capsys, given):
    monkeypatch.setenv("CLM_PORTAL_KEY", KEY)
    ran = []
    assert clm_portal.serve(serve_args(environment, "--public-origin", given), run=lambda *a, **k: ran.append(a)) == 2
    err = capsys.readouterr().err
    assert "--public-origin" in err and len(err.strip().splitlines()) == 1 and not ran


def test_serve_passes_the_public_origin_to_the_page(environment, monkeypatch):
    monkeypatch.setenv("CLM_PORTAL_KEY", KEY)
    apps = []
    assert clm_portal.serve(serve_args(environment, "--public-origin", PUBLIC + "/"),
                            run=lambda app, **kwargs: apps.append(app)) == 0
    assert apps[0].public == PUBLIC and apps[0].hosts == {"127.0.0.1:8400", "localhost:8400", "portal.example"}


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


def test_serve_refuses_a_short_key(environment, monkeypatch, capsys):
    ran = []
    monkeypatch.setenv("CLM_PORTAL_KEY", "x" * 15)
    assert clm_portal.serve(serve_args(environment), run=lambda *a, **k: ran.append(a)) == 2
    err = capsys.readouterr().err
    assert "at least 16 characters" in err and "x" * 15 not in err and len(err.strip().splitlines()) == 1
    assert not ran
    monkeypatch.setenv("CLM_PORTAL_KEY", "x" * 16)
    assert clm_portal.serve(serve_args(environment), run=lambda *a, **k: ran.append(a)) == 0 and ran


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


def serving_without_the_handbook(left_out: list[str]) -> MCPServer:
    server = MCPServer("stand-in")

    @server.tool()
    def describe() -> dict:
        """What the environment offers, with no collection, and these tools left out by [serve]."""
        return {"name": "Tiny handbook", **({"left_out": left_out} if left_out else {})}
    return server


def test_a_collection_whose_tools_serve_leaves_out_says_so(client, server):
    sign_in(client)
    server.server = serving_without_the_handbook(["search_handbook", "document_handbook", "decision_vesting"])
    for path in ("/", "/documents"):
        page = client.get(path).text
        assert "not offered: the server's [serve] setting leaves out its tools" in page, path
        assert "not known" not in page, path
    for left_out in ([], ["decision_vesting"]):
        server.server = serving_without_the_handbook(left_out)
        page = client.get("/").text
        assert "not known" in page and "not offered" not in page


HOSTILE = "<script>alert(1)</script>"


def test_every_name_and_description_from_the_files_is_escaped(client, environment):
    text = environment.read_text().replace('name = "Tiny handbook"', f'name = "{HOSTILE}"')
    for old in ('"A made-up handbook, for the portal\'s tests."', '"The made-up handbook."',
                '"How much of the match is the employee\'s."', '"Full years of service."'):
        text = text.replace(old, f'"{HOSTILE}"')
    text += f'\n[information."{HOSTILE}"]\ndescription = "{HOSTILE}"\nkind = "supplied"\n'
    text += f'\n[collections."{HOSTILE}"]\ndescription = "{HOSTILE}"\n'
    environment.write_text(text)
    file = environment.parent / "derived" / "derived.json"
    found = json.loads(file.read_text())
    found["facts"][f"{HOSTILE}.cap"] = {**found["facts"]["k401_2026.match_cap"], "node": HOSTILE}
    found["decisions"]["carryover_limit"]["description"] = HOSTILE
    file.write_text(json.dumps(found))
    sign_in(client)
    for path in ("/", "/review", "/documents"):
        page = client.get(path).text
        assert "<script>" not in page and html.escape(HOSTILE) in page, path


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


class EchoesItsKey(FakePopen):
    """A clm-server that prints the key it was given, then exits: the worst its error output could do."""

    def __call__(self, argv, *, env, stderr, **kwargs):
        self.stderr = f"error: cannot start with the key {env[managed.SERVER_KEY_ENV]}\n"
        return super().__call__(argv, env=env, stderr=stderr, **kwargs)


@pytest.mark.parametrize("path", PAGES + ["/sign-in"])
def test_the_keys_never_appear_on_a_page(environment, portal, path, monkeypatch):
    monkeypatch.setattr(web, "WRONG_KEY_WAIT", 0.01)
    monkeypatch.setenv("CLM_PORTAL_KEY", KEY)
    monkeypatch.setenv("LLM_API_KEY", "the-model-key-not-a-secret")
    popen = EchoesItsKey(code=2)
    portal.server = managed.ManagedServer(environment, popen=popen, connect=lambda url, key: None, wait=0.5,
                                          find=lambda name: Path("/stand-in/bin") / name)
    with TestClient(web.create_app(portal, KEY, "127.0.0.1", 8400), base_url=ORIGIN) as client:
        server_key = popen.started[0].env[managed.SERVER_KEY_ENV]
        keys = (KEY, server_key, "the-model-key-not-a-secret")
        assert len(server_key) >= 32 and server_key in popen.stderr
        assert not any(key in client.get("/sign-in").text for key in keys)
        wrong = client.post("/sign-in", data={"key": KEY + "x", "name": "K"}, headers={"Origin": ORIGIN}).text
        assert not any(key in wrong for key in keys)
        signed = client.post("/sign-in", data={"key": KEY, "name": "Kevin"}, headers={"Origin": ORIGIN}).text
        assert "cannot start with the key ***" in signed and not any(key in signed for key in keys)
        page = client.get(path).text
        assert not any(key in page for key in keys)
