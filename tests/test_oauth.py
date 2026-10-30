"""Tests for oauth.py and `clm-server --auth oauth`: tokens from an OIDC issuer, checked as JWTs."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
import asyncio
import json
import time

import httpx2
import jwt
import pytest
from conftest import ENV_TOML, free_port, serving
from cryptography.hazmat.primitives.asymmetric import rsa
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from test_server import parts  # noqa: F401 pylint: disable=unused-import

from clm_decision import oauth
from clm_decision import server
from clm_decision.environment import load_environment
from clm_decision.environment_server import build_environment_server
from clm_decision import tracing

ISSUER = "https://login.example.test/realms/clm"
AUDIENCE = "decision-server"


def _key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


KEY, OTHER_KEY = _key(), _key()


class Issuer:
    """The issuer's discovery document and JWKS, served through `fetch`; counts each fetch."""

    def __init__(self, issuer=ISSUER, keys=((KEY, "k1"),)):
        self.issuer, self.fetched = issuer, []
        self.jwks = {"keys": [{**jwt.algorithms.RSAAlgorithm.to_jwk(key.public_key(), as_dict=True), "kid": kid,
                               "use": "sig", "alg": "RS256"} for key, kid in keys]}

    async def fetch(self, url):
        self.fetched.append(url)
        if url == f"{ISSUER}/.well-known/openid-configuration":
            return {"issuer": self.issuer, "jwks_uri": f"{ISSUER}/certs"}
        if url == f"{ISSUER}/certs":
            return self.jwks
        raise oauth.OAuthError(f"no such url {url}")


def token(key=KEY, kid="k1", algorithm="RS256", **claims):
    now = int(time.time())
    body = {"iss": ISSUER, "aud": AUDIENCE, "sub": "user-1", "azp": "databricks", "iat": now, "exp": now + 300,
            "scope": "openid decide"}
    body.update(claims)
    return jwt.encode({key: value for key, value in body.items() if value is not None}, key, algorithm=algorithm,
                      headers={"kid": kid})


def verifier(issuer=None):
    return oauth.OidcVerifier(ISSUER, AUDIENCE, fetch=(issuer or Issuer()).fetch)


@pytest.mark.anyio
async def test_a_good_token_is_accepted():
    access = await verifier().verify_token(token())
    assert (access.client_id, access.subject, access.resource) == ("databricks", "user-1", AUDIENCE)
    assert "decide" in access.scopes


@pytest.mark.anyio
@pytest.mark.parametrize("claims", [{"exp": int(time.time()) - 120}, {"aud": "someone-else"},
                                    {"iss": "https://elsewhere.test"}, {"exp": None}, {"aud": None},
                                    {"nbf": int(time.time()) + 600}])
async def test_a_bad_claim_is_refused(claims):
    assert await verifier().verify_token(token(**claims)) is None


@pytest.mark.anyio
async def test_an_unknown_signing_key_is_refused():
    assert await verifier().verify_token(token(key=OTHER_KEY)) is None
    assert await verifier().verify_token(token(key=OTHER_KEY, kid="k2")) is None


@pytest.mark.anyio
async def test_a_symmetric_token_is_refused():
    forged = jwt.encode({"iss": ISSUER, "aud": AUDIENCE, "exp": int(time.time()) + 300}, "secret" * 8,
                        algorithm="HS256", headers={"kid": "k1"})
    assert await verifier().verify_token(forged) is None
    assert await verifier().verify_token("not a token") is None


@pytest.mark.anyio
@pytest.mark.parametrize("claims, scopes", [({"scope": "decide openid"}, {"decide", "openid"}),
                                            ({"scope": None, "scp": "decide"}, {"decide"}),
                                            ({"scope": None, "roles": ["decide", "other"]}, {"decide", "other"})])
async def test_scopes_come_from_scope_scp_or_roles(claims, scopes):
    assert set((await verifier().verify_token(token(**claims))).scopes) == scopes


@pytest.mark.anyio
async def test_entra_app_only_tokens_name_the_client_by_appid():
    access = await verifier().verify_token(token(azp=None, appid="app-guid"))
    assert access.client_id == "app-guid"


@pytest.mark.anyio
async def test_keys_are_fetched_once_and_again_for_a_new_kid(monkeypatch):
    monkeypatch.setattr(oauth, "REFETCH_SECONDS", 0)
    issuer = Issuer(keys=((KEY, "k1"),))
    check = verifier(issuer)
    await check.verify_token(token())
    await check.verify_token(token())
    assert len(issuer.fetched) == 2
    issuer.jwks["keys"].append({**jwt.algorithms.RSAAlgorithm.to_jwk(OTHER_KEY.public_key(), as_dict=True),
                                "kid": "k2", "alg": "RS256"})
    assert await check.verify_token(token(key=OTHER_KEY, kid="k2")) is not None
    assert issuer.fetched.count(f"{ISSUER}/certs") == 2


@pytest.mark.anyio
async def test_unknown_kids_refetch_at_most_once_a_minute():
    issuer = Issuer()
    check = verifier(issuer)
    await check.verify_token(token())
    for kid in ("x1", "x2", "x3"):
        assert await check.verify_token(token(key=OTHER_KEY, kid=kid)) is None
    assert issuer.fetched.count(f"{ISSUER}/certs") == 1


@pytest.mark.anyio
async def test_a_discovery_document_for_another_issuer_is_refused():
    assert await verifier(Issuer(issuer="https://evil.test")).verify_token(token()) is None


@pytest.fixture
def oauth_url(parts, tmp_path):
    """A server on a free port that takes tokens from the test issuer; -> (its MCP address, its trace)."""
    port = free_port()
    url = f"http://127.0.0.1:{port}/mcp"
    auth = oauth.oauth(ISSUER, AUDIENCE, ["decide"], url, fetch=Issuer().fetch)
    trace = tracing.TraceWriter(tmp_path / "trace.jsonl")
    with serving(server.http_app(server.build_server(*parts, review=False, trace=trace, auth=auth), None), port):
        yield url, trace


@pytest.mark.anyio
async def test_a_token_decides_over_http_and_the_trace_names_the_caller(oauth_url):
    url, trace = oauth_url
    async with httpx2.AsyncClient(headers={"Authorization": f"Bearer {token()}"}) as http:
        async with Client(streamable_http_client(url, http_client=http)) as client:
            result = await client.call_tool("decide", {"text": "My laptop will not start"})
    assert result.structured_content["route"]
    trace.close()
    called = [json.loads(line) for line in trace.path.read_text(encoding="utf-8").splitlines()]
    called = [record for record in called if record["tool"] == "decide"]
    assert called[0]["caller"] == {"client_id": "databricks", "subject": "user-1"}
    assert token().split(".", maxsplit=1)[0] not in json.dumps(called[0])


@pytest.mark.parametrize("headers, status", [({}, 401), ({"Authorization": "Bearer k1"}, 401)])
def test_http_without_a_good_token_is_401(oauth_url, headers, status):
    response = httpx2.post(oauth_url[0], headers=headers, json={})
    assert response.status_code == status and "WWW-Authenticate" in response.headers


def test_http_without_the_scope_is_403(oauth_url):
    response = httpx2.post(oauth_url[0], headers={"Authorization": f"Bearer {token(scope='openid')}"}, json={})
    assert response.status_code == 403


def test_the_protected_resource_metadata_names_the_issuer(oauth_url):
    base = oauth_url[0].removesuffix("/mcp")
    metadata = httpx2.get(f"{base}/.well-known/oauth-protected-resource/mcp").json()
    assert metadata["authorization_servers"] == [ISSUER] and metadata["resource"] == oauth_url[0]


def test_oauth_needs_an_issuer(write_scenario, monkeypatch, capsys):
    monkeypatch.delenv("DECISION_OAUTH_ISSUER", raising=False)
    assert server.main([str(write_scenario()), "--http", "--auth", "oauth", "--audience", AUDIENCE]) == 2
    assert "--auth oauth needs --issuer or DECISION_OAUTH_ISSUER" in capsys.readouterr().err


@pytest.mark.anyio
async def test_an_environment_server_takes_tokens_too(write_environment):
    port = free_port()
    url = f"http://127.0.0.1:{port}/mcp"
    auth = oauth.oauth(ISSUER, AUDIENCE, ["decide"], url, fetch=Issuer().fetch)
    env_server = build_environment_server(load_environment(write_environment(ENV_TOML)), auth=auth)
    with serving(server.http_app(env_server, None), port):
        assert httpx2.post(url, json={}).status_code == 401
        async with httpx2.AsyncClient(headers={"Authorization": f"Bearer {token()}"}) as http:
            async with Client(streamable_http_client(url, http_client=http)) as client:
                described = await client.call_tool("describe", {})
    assert described.structured_content["name"]


@pytest.mark.anyio
async def test_a_failing_issuer_is_asked_at_most_once_a_minute():
    asked = []

    async def down(url):
        asked.append(url)
        raise oauth.OAuthError(f"cannot read {url}: connection refused")
    check = oauth.OidcVerifier(ISSUER, AUDIENCE, fetch=down)
    for kid in ("k1", "x1", "x2", "k1"):
        assert await check.verify_token(token(kid=kid)) is None
    assert len(asked) == 1


@pytest.mark.anyio
async def test_a_known_key_does_not_wait_for_a_slow_fetch(monkeypatch):
    issuer, release = Issuer(), asyncio.Event()
    check = verifier(issuer)
    await check.verify_token(token())
    monkeypatch.setattr(oauth, "REFETCH_SECONDS", 0)

    async def slow(url):
        await release.wait()
        return await issuer.fetch(url)
    check.fetch = slow
    stalled = asyncio.create_task(check.verify_token(token(key=OTHER_KEY, kid="junk")))
    await asyncio.sleep(0.05)
    assert await asyncio.wait_for(check.verify_token(token()), timeout=1) is not None
    release.set()
    assert await stalled is None


@pytest.mark.anyio
async def test_odd_claims_and_documents_are_refused_or_mended_not_errors():
    access = await verifier().verify_token(token(exp=time.time() + 300.5))
    assert isinstance(access.expires_at, int)
    assert await verifier().verify_token(token(sub=12345)) is None

    async def listing(_url):
        return ["not", "a", "document"]
    assert await oauth.OidcVerifier(ISSUER, AUDIENCE, fetch=listing).verify_token(token()) is None


@pytest.mark.anyio
async def test_alg_none_and_kidless_tokens():
    unsigned = jwt.encode({"iss": ISSUER, "aud": AUDIENCE, "exp": int(time.time()) + 300}, None, algorithm="none")
    assert await verifier().verify_token(unsigned) is None
    kidless = jwt.encode({"iss": ISSUER, "aud": [AUDIENCE, "account"], "exp": int(time.time()) + 300}, KEY,
                         algorithm="RS256")
    assert await verifier().verify_token(kidless) is not None
    assert await verifier(Issuer(keys=((KEY, "k1"), (OTHER_KEY, "k2")))).verify_token(kidless) is None
