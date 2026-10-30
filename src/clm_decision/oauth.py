"""OAuth for the HTTP server: tokens from any OIDC issuer, such as Entra ID or Keycloak, checked as JWTs.

The server is an OAuth resource server, as MCP's authorization spec describes: it checks tokens and never issues
them. The issuer's signing keys are found through its discovery document and kept, and fetched again when a token
names a key not seen yet. The MCP SDK turns a refused token into 401, a missing scope into 403, and serves the
protected resource metadata that tells a client which issuer to ask.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

import httpx
import jwt
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings

# Signatures by the issuer's public keys only: never "none", never a shared secret
ALGORITHMS = ["RS256", "RS384", "RS512", "PS256", "PS384", "PS512", "ES256", "ES384", "ES512"]
LEEWAY_SECONDS = 30
# A token naming an unknown key fetches the keys again, but at most this often
REFETCH_SECONDS = 60

ISSUER_ENV, AUDIENCE_ENV, SCOPES_ENV = "DECISION_OAUTH_ISSUER", "DECISION_OAUTH_AUDIENCE", "DECISION_OAUTH_SCOPES"

log = logging.getLogger("decision-server.oauth")


class OAuthError(Exception):
    """The issuer could not be reached, or answered with something other than its documents."""


async def fetch_json(url: str) -> dict:
    """GET a JSON document; raises OAuthError in one line."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()
    except (httpx.HTTPError, ValueError) as error:
        raise OAuthError(f"cannot read {url}: {error}") from error


def _scopes(claims: dict) -> list[str]:
    """RFC 9068 and Keycloak use `scope`, Entra's delegated tokens `scp`: space-separated. Entra's app-only tokens
    carry app roles in `roles`, a list."""
    scopes = []
    for name in ("scope", "scp"):
        if isinstance(claims.get(name), str):
            scopes += claims[name].split()
    if isinstance(claims.get("roles"), list):
        scopes += [role for role in claims["roles"] if isinstance(role, str)]
    return list(dict.fromkeys(scopes))


class OidcVerifier:
    """Checks a token's signature, issuer, audience, expiry and not-before; the SDK then checks the scopes."""

    def __init__(self, issuer: str, audience: str, fetch: Callable[[str], Awaitable[dict]] = fetch_json):
        self.issuer, self.audience, self.fetch = issuer, audience, fetch
        self._jwks_uri: str | None = None
        self._keys: jwt.PyJWKSet | None = None
        # When keys were last asked for, whether or not the issuer answered
        self._asked: float | None = None
        self._lock = asyncio.Lock()

    def _may_ask(self) -> bool:
        return self._asked is None or time.monotonic() - self._asked > REFETCH_SECONDS

    async def _key(self, kid: str | None) -> jwt.PyJWK | None:
        # A known key, or a refusal while asking again is not allowed, never waits for a fetch in progress
        key = self._find(kid)
        if key is not None or not self._may_ask():
            return key
        async with self._lock:
            if self._may_ask():
                self._asked = time.monotonic()
                await self._fetch_keys()
            return self._find(kid)

    def _find(self, kid: str | None) -> jwt.PyJWK | None:
        if self._keys is None:
            return None
        keys = self._keys.keys
        return next((key for key in keys if key.key_id == kid), keys[0] if kid is None and len(keys) == 1 else None)

    async def _document(self, url: str) -> dict:
        document = await self.fetch(url)
        if not isinstance(document, dict):
            raise OAuthError(f"{url} did not answer with a JSON object")
        return document

    async def _fetch_keys(self) -> None:
        if self._jwks_uri is None:
            document = await self._document(self.issuer.rstrip("/") + "/.well-known/openid-configuration")
            if document.get("issuer") != self.issuer:
                raise OAuthError(f"the discovery document names issuer {document.get('issuer')!r}, "
                                 f"not {self.issuer!r}")
            self._jwks_uri = str(document["jwks_uri"])
        self._keys = jwt.PyJWKSet.from_dict(await self._document(self._jwks_uri))

    async def verify_token(self, token: str) -> AccessToken | None:
        """The token's access, or None to refuse it."""
        try:
            key = await self._key(jwt.get_unverified_header(token).get("kid"))
            if key is None:
                return None
            claims = jwt.decode(token, key.key, algorithms=ALGORITHMS, audience=self.audience, issuer=self.issuer,
                                leeway=LEEWAY_SECONDS, options={"require": ["exp", "iss", "aud"]})
        except jwt.PyJWTError:
            return None
        except (OAuthError, KeyError) as error:
            log.warning("oauth: cannot check tokens: %s", error)
            return None
        # Anything else unexpected refuses the token too: an error here would be a 500, not a 401
        except Exception as error:  # pylint: disable=broad-exception-caught
            log.warning("oauth: refused a token: %s: %s", type(error).__name__, error)
            return None
        # azp: OIDC and Entra v2; appid: Entra v1; client_id: RFC 9068 and Keycloak's client credentials
        client = claims.get("client_id") or claims.get("azp") or claims.get("appid") or "unknown"
        subject = claims.get("sub")
        return AccessToken(token=token, client_id=str(client), scopes=_scopes(claims), expires_at=int(claims["exp"]),
                           resource=self.audience, subject=None if subject is None else str(subject),
                           claims={"iss": claims["iss"]})


@dataclass(frozen=True)
class OAuth:
    """What MCPServer needs to take tokens: its auth settings and the verifier."""
    settings: AuthSettings
    verifier: OidcVerifier


def oauth(issuer: str, audience: str, scopes: list[str], resource_url: str,
          fetch: Callable[[str], Awaitable[dict]] = fetch_json) -> OAuth:
    """OAuth for a server at `resource_url`, taking tokens `issuer` made for `audience` with every one of `scopes`."""
    # The verifier checks the audience, which need not be the URL: Entra's is often api://<app id>
    return OAuth(AuthSettings(issuer_url=issuer, resource_server_url=resource_url, required_scopes=scopes or None,
                              validate_token_resource=False),
                 OidcVerifier(issuer, audience, fetch))


def from_args(issuer: str | None, audience: str | None, scopes: list[str] | None, resource_url: str) -> OAuth:
    """OAuth from the command line, each setting falling back to its variable; raises ValueError naming what is
    missing."""
    issuer = issuer or os.environ.get(ISSUER_ENV, "").strip()
    if not issuer:
        raise ValueError(f"--auth oauth needs --issuer or {ISSUER_ENV}")
    if not issuer.startswith(("https://", "http://localhost", "http://127.0.0.1")):
        raise ValueError(f"--issuer must be https (or http on localhost), not {issuer}")
    audience = audience or os.environ.get(AUDIENCE_ENV, "").strip() or resource_url
    scopes = scopes if scopes is not None else os.environ.get(SCOPES_ENV, "").split()
    return oauth(issuer, audience, scopes, resource_url)


def server_options(auth: OAuth | None) -> dict:
    """MCPServer's keyword arguments for `auth`: none without it."""
    return {"auth": auth.settings, "token_verifier": auth.verifier} if auth else {}
