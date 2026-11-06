"""The Server page: what the running server is, its checks, its tools and the credentials it uses.

The operator can run the checks now and disable or enable a tool; a change reaches agents at once, but a client may
keep the tool list it fetched for up to 30 seconds (and longer, if it is an older client that ignores the hint).
No credential is ever shown: a key by its 8-hex id, a password or the portal key only as `set` or `missing`.
"""
from __future__ import annotations

import importlib.metadata
import os
import time

from mcp.types import LATEST_PROTOCOL_VERSION
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response
from starlette.routing import Route

from clm_decision.environment_server import infrastructure

from . import chat
from .web import csrf_field, esc, page, table

STALE_WINDOW = ("When a tool is disabled or enabled, agents are told the list changed, but a client may keep the "
                "tool list it fetched for up to 30 seconds, and longer if it is an older client that ignores the "
                "hint. A disabled tool is refused as soon as it is called.")


def when(moment: float | None) -> str:
    """A time of day and date, or a dash for none."""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(moment)) if moment else "-"


def present(variable: str) -> str:
    """Whether the environment variable has a value, as `set` or `missing`; never the value."""
    return "set" if os.environ.get(variable) else "missing"


def state_rows(portal) -> list[list[str]]:
    """The server's name, files, versions, mode and tool counts."""
    live = portal.server
    offering, env = live.server.offering, live.server.holder.current
    checks = live.checks
    left_out = sorted(set(infrastructure(env)) - set(offering.offered))
    held = sorted(live.admin.hidden()) if live.admin else []
    rows = [("Name", env.name), ("Environment file", portal.environment),
            ("Package version", importlib.metadata.version("clm-decision-agent")),
            ("MCP protocol version", LATEST_PROTOCOL_VERSION), ("Started", when(portal.started_at)),
            ("Read only", "yes" if offering.serve.read_only else "no"),
            ("Authentication", "OAuth" if portal.oauth else "key"),
            ("Tool list fingerprint", checks.baseline[:16] if checks else "-"),
            ("Offered tools", ", ".join(offering.offered) or "-"),
            ("Left out of the offer", ", ".join(left_out) or "-"),
            ("Disabled by the operator", ", ".join(held) or "-")]
    return [[esc(label), esc(value)] for label, value in rows]


def checks_section(portal, session) -> str:
    """The timer's latest checks and the run-now form."""
    checks = portal.server.checks
    rows = [[esc(check.name), f'<span class="{esc(check.status.replace(" ", "-"))}">{esc(check.status)}</span>',
             esc(check.detail), esc(when(check.measured_at)), esc(when(check.last_success)),
             esc(check.last_error or "-"), esc(", ".join(check.tools) or "-")]
            for check in (checks.latest if checks else [])]
    body = table(["Check", "Status", "Detail", "Measured", "Last success", "Last error", "Tools affected"], rows) \
        if rows else "<p>No check has run yet.</p>"
    return (f"<h2>Checks</h2>{body}<form method=\"post\" action=\"/server/checks\" class=\"inline\">"
            f"{csrf_field(session)}<button>Run checks now</button></form>")


def tool_form(session, name: str, disabled: bool) -> str:
    """The Enable or Disable form of one tool."""
    action = "enable" if disabled else "disable"
    return (f'<form method="post" action="/server/tools/{esc(name)}/{action}" class="inline">{csrf_field(session)}'
            f'<button>{action.capitalize()}</button></form>')


def tools_section(portal, session) -> str:
    """Each offered tool, with whether it is read only, enabled, what it runs on and its switch."""
    live = portal.server
    offering, env = live.server.offering, live.server.holder.current
    registered = {tool.name: tool for tool in live.server._tool_manager.list_tools()}  # pylint: disable=protected-access
    uses = infrastructure(env)
    held = live.admin.disabled if live.admin else frozenset()
    rows = []
    for name in offering.offered:
        annotations = registered[name].annotations if name in registered else None
        read_only = bool(annotations and annotations.read_only_hint)
        rows.append([esc(name), "yes" if read_only else "no", "disabled" if name in held else "enabled",
                     esc(", ".join(sorted(uses.get(name, ()))) or "-"),
                     "" if name == "describe" or live.admin is None else tool_form(session, name, name in held)])
    return (f"<h2>Tools</h2><p>{STALE_WINDOW}</p>"
            + table(["Tool", "Read only", "State", "Infrastructure", ""], rows))


def credentials_section(portal) -> str:
    """The credentials the server uses, each as an id, `set` or `missing`, or (OAuth) the settings in full."""
    rows = []
    if portal.keys is not None:
        used = portal.keys.last_used
        rows += [[f"Agents' key {esc(ident)}", f"last used {esc(when(used[ident]) if ident in used else 'never')}"]
                 for ident in portal.keys.ids()]
    if portal.oauth is not None:
        settings = portal.oauth.settings
        rows += [["OAuth issuer", esc(settings.issuer_url)], ["OAuth audience", esc(portal.oauth.verifier.audience)],
                 ["OAuth scopes", esc(" ".join(settings.required_scopes or ()) or "none")]]
    env = portal.server.server.holder.current
    rows.append(["Portal key", "set"])
    if env.document_settings is not None:
        rows.append([f"PostgreSQL password ({esc(env.document_settings.password_env)})",
                     present(env.document_settings.password_env)])
    if env.graph_settings is not None and env.graph_settings.backend == "neo4j":
        rows.append([f"Neo4j password ({esc(env.graph_settings.password_env)})",
                     present(env.graph_settings.password_env)])
    rows.append([f"Model key ({chat.LLM_KEY_ENV})", present(chat.LLM_KEY_ENV)])
    return "<h2>Credentials</h2>" + table(["Credential", "State"], rows)


def server_body(portal, session, message: str = "") -> str:
    """The whole page's body; `message` is shown as an error above it."""
    shown = f'<p class="failed">{esc(message)}</p>' if message else ""
    return (shown + table(["", ""], state_rows(portal)) + checks_section(portal, session)
            + tools_section(portal, session) + credentials_section(portal))


def refused(request: Request, message: str, status: int) -> Response:
    """The page again with `message` as the error."""
    portal, session = request.app.state.portal, request.state.session
    return page(portal, session, "Server", server_body(portal, session, message), status=status)


async def server_page(request: Request) -> Response:
    """GET /server."""
    portal, session = request.app.state.portal, request.state.session
    return page(portal, session, "Server", server_body(portal, session))


async def run_checks_now(request: Request) -> Response:
    """POST /server/checks: run every check now, the OAuth keys fetched again."""
    checks = request.app.state.portal.server.checks
    if checks is None:
        return refused(request, "The server has no checks.", 409)
    await checks.run(refresh=True)
    return RedirectResponse("/server", status_code=303)


async def switch_tool(request: Request) -> Response:
    """POST /server/tools/{name}/disable and /enable. The registry changes on the event loop, then agents are told."""
    admin = request.app.state.portal.server.admin
    if admin is None:
        return refused(request, "This server cannot disable tools.", 409)
    name, action = request.path_params["name"], request.path_params["action"]
    if action not in ("disable", "enable"):
        return refused(request, f"No action {action}: a tool is switched by disable or enable.", 404)
    try:
        (admin.disable if action == "disable" else admin.enable)(name)
    except ValueError as error:
        return refused(request, str(error), 400)
    await admin.changed()
    return RedirectResponse("/server", status_code=303)


def routes() -> list:
    """The Server page's routes."""
    return [Route("/server", server_page), Route("/server/checks", run_checks_now, methods=["POST"]),
            Route("/server/tools/{name}/{action:str}", switch_tool, methods=["POST"])]
