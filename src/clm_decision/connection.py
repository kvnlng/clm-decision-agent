"""How a client reaches a decision server: a file is started over stdio, as an agent would start it; a URL is reached
over HTTP, sending the key in DECISION_SERVER_KEY when it is set. clm-agent uses it; to call tools by hand, use the
MCP Inspector (docs/getting-started.md, "Looking inside the server").
"""
from __future__ import annotations

import contextlib
import os
import sys

from mcp import StdioServerParameters
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client

KEY_ENV = "DECISION_SERVER_KEY"


def is_url(target: str) -> bool:
    """Whether the target is a running server's address rather than a file."""
    return target.startswith(("http://", "https://"))


def server_params(path: str) -> StdioServerParameters:
    """How to start the decision server for a file. It gets this process's variables, so a password the
    environment names (NEO4J_PASSWORD) reaches it; the MCP client would otherwise pass on only a few, such as PATH."""
    return StdioServerParameters(command=sys.executable, args=["-m", "clm_decision.server", path],
                                 env=dict(os.environ))


@contextlib.asynccontextmanager
async def _http(url: str, refused: list[int]):
    key = os.environ.get(KEY_ENV)
    headers = {"Authorization": f"Bearer {key}"} if key else None
    async with create_mcp_http_client(headers=headers) as http, \
            streamable_http_client(url, http_client=http) as streams:
        # The SDK reports a refusal only as "an error response"; keep its status to say why
        async def note(response) -> None:
            if response.status_code in (401, 403):
                refused.append(response.status_code)
        http.event_hooks["response"].append(note)
        yield streams


def connect(target: str, refused: list[int] | None = None):
    """What to give mcp.Client for a file (started over stdio) or a URL (over HTTP, with the key if set). A list
    given as `refused` collects the status of any response refusing the key."""
    return _http(target, refused if refused is not None else []) if is_url(target) else server_params(target)


def refusal(refused: list[int]) -> str | None:
    """Why the server refused the caller, if it did."""
    if not refused:
        return None
    sent = "with no key" if not os.environ.get(KEY_ENV) else f"with the key in {KEY_ENV}"
    return f"the server refused the call ({refused[0]}), {sent}"

