"""The portal's page: sign-in, then Configuration, Ask, Review and Documents, as plain HTML from Python.

Every request must carry the bound address in its Host header, which stops DNS rebinding. Every POST must come from
the portal's own origin and, once signed in, carry the session's CSRF token. Sessions are held in memory, so a
restart signs everyone out.
"""
from __future__ import annotations

import asyncio
import contextlib
import hmac
import html
import json
import secrets
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, PlainTextResponse, RedirectResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from . import layer
from .managed import Jobs, ToolFailed, script

COOKIE = "clm_portal_session"
SESSION_SECONDS = 12 * 60 * 60
WRONG_KEY_WAIT = 0.5
MAX_BODY = 200 * 1024 * 1024
STATIC = Path(__file__).parent / "static"
OPEN_PATHS = ("/sign-in",)
OPERATOR_NOTE = ("The portal restarts its own server after every change. The server agents use reads the files at "
                 "start: its operator restarts it, or redeploys from git, to pick up approved changes.")
TESTING_BANNER = "Testing: unapproved derived rules are deciding."


@dataclass
class Session:
    """A signed-in person: the name they gave, the token every form carries, and their conversation."""
    name: str
    csrf: str
    expires: float
    ask: dict = field(default_factory=dict)


class Portal:  # pylint: disable=too-few-public-methods
    """What the page works on: the environment file, the portal's server, its jobs and the signed-in sessions."""

    def __init__(self, environment: Path, server, jobs: Jobs, find=script, options: dict | None = None):
        self.environment, self.server, self.jobs, self.find = Path(environment), server, jobs, find
        self.options = options or {}
        self.sessions: dict[str, Session] = {}

    def session(self, ident: str | None) -> Session | None:
        """The live session with this id, if any."""
        found = self.sessions.get(ident or "")
        if found is not None and found.expires < time.time():
            del self.sessions[ident]
            return None
        return found


def esc(value: Any) -> str:
    """Text for HTML."""
    return html.escape("" if value is None else str(value))


def shown_value(value: Any) -> str:
    """A value as JSON when it is not plain text."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def csrf_field(session: Session) -> str:
    """The hidden field every form carries."""
    return f'<input type="hidden" name="csrf" value="{esc(session.csrf)}">'


# pylint: disable-next=too-many-arguments
def page(portal: Portal, session: Session | None, title: str, body: str, *, status: int = 200,
         changes: bool = False) -> HTMLResponse:
    """A whole page: the sections, the testing banner when it is on, and sign-out."""
    if session is None:
        nav = ""
    else:
        links = "".join(f'<a href="{href}">{name}</a>' for href, name in (
            ("/", "Configuration"), ("/ask", "Ask"), ("/review", "Review"), ("/documents", "Documents")))
        nav = (f'<nav>{links}<span class="who">Signed in as {esc(session.name)}</span>'
               f'<form method="post" action="/sign-out">{csrf_field(session)}<button>Sign out</button></form></nav>')
    banner = f'<p class="banner testing">{TESTING_BANNER}</p>' if portal.server.testing else ""
    note = f'<p class="note">{OPERATOR_NOTE}</p>' if changes else ""
    return HTMLResponse(
        f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width, initial-scale=1"><title>{esc(title)} - portal</title>'
        f'<link rel="stylesheet" href="/static/portal.css"><script src="/static/portal.js" defer></script></head>'
        f'<body>{nav}{banner}<main><h1>{esc(title)}</h1>{note}{body}</main></body></html>', status_code=status)


def table(columns: list[str], rows: list[list[str]]) -> str:
    """A table of already escaped cells."""
    head = "".join(f"<th>{esc(column)}</th>" for column in columns)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def passage_link(collection: str | None, document: str, passage: str | None = None) -> str:
    """A link to a cited passage, read through the portal's server."""
    if not collection:
        return esc(document if passage is None else f"{document} {passage}")
    query = {"collection": collection, "id": document, **({"passage": passage} if passage else {})}
    label = document if passage is None else f"{document} {passage}"
    return f'<a href="/passage?{esc(urllib.parse.urlencode(query))}">{esc(label)}</a>'


def dmn_table(rows: dict | None) -> str:
    """A decision table's rows."""
    if rows is None:
        return '<p class="muted">No table could be read.</p>'
    return table(["row", *rows["inputs"], *rows["outputs"]],
                 [[esc(row["id"]), *(esc(cell) for cell in row["when"]),
                   *(esc(shown_value(cell)) for cell in row["then"])] for row in rows["rows"]])


def check_text(check: dict | None) -> str:
    """A check, passed or with its failures."""
    if check is None:
        return '<span class="muted">not checked yet</span>'
    if check.get("passed"):
        return '<span class="passed">check passed</span>'
    failures = "".join(f"<li>{esc(item)}</li>" for item in check.get("failures") or [])
    return f'<span class="failed">check failed</span><ul>{failures}</ul>'


def source_links(collection: str | None, source: dict | None) -> str:
    """A derived entry's cited passages, each a link, and its quote."""
    if not source:
        return ""
    document = source.get("document", "")
    links = ", ".join(passage_link(collection, document, item) for item in source.get("passages") or []) \
        or passage_link(collection, document)
    quote = f'<blockquote>{esc(source.get("quote"))}</blockquote>' if source.get("quote") else ""
    return f'<p>Source: {links}{" (edition " + esc(source["edition"]) + ")" if source.get("edition") else ""}</p>' \
           f'{quote}'


def _rule(rule: dict) -> str:
    about = rule["rule"]
    origin = f'<span class="origin {esc(rule["origin"])}">{esc(rule["origin"])}</span>'
    status = f' <span class="status {esc(rule["status"])}">{esc(rule["status"])}</span>' if rule["status"] else ""
    facts = "".join(f"<dt>{esc(key)}</dt><dd>{esc(shown_value(about[key]))}</dd>"
                    for key in ("statement", "reason", "owner", "version", "governs", "depends_on", "needs")
                    if about.get(key))
    cites = ", ".join(passage_link(rule["collection"], cited) for cited in about.get("cites") or [])
    other = "".join(f"<dt>{esc(key)}</dt><dd>{esc(shown_value(value))}</dd>" for key, value in rule["other"].items())
    body = (f'<h3>{esc(rule["name"])} {origin}{status}</h3><p>{esc(rule["description"])}</p><dl>{facts}'
            f'{"<dt>cites</dt><dd>" + cites + "</dd>" if cites else ""}{other}</dl>')
    if rule["origin"] == "derived":
        body += f'<p>{check_text(rule["check"])}</p>{source_links(rule["collection"], rule["source"])}'
    if rule["function"]:
        body += f'<p>Calculated by the function <code>{esc(rule["function"])}</code>.</p>'
    elif rule["dmn"]:
        body += f'<p><code>{esc(rule["dmn"])}</code></p>{dmn_table(rule["table"])}'
    return f'<section class="rule">{body}</section>'


def configuration_body(found: dict, running: bool) -> str:
    """The Configuration page's sections."""
    parts = [f'<p>{esc(found["description"])}</p>']
    if found["errors"]:
        items = "".join(f"<li><pre>{esc(error)}</pre></li>" for error in found["errors"])
        parts.append(f'<section class="errors"><h2>Load errors</h2><ul>{items}</ul></section>')
    if running:
        tools = table(["tool", "description", "parameters"],
                      [[f"<code>{esc(tool['name'])}</code>", esc(tool["description"]),
                        esc(", ".join(tool["parameters"]))] for tool in found["tools"]])
    else:
        tools = '<p class="muted">The portal\'s server is not running, so its tools are not listed.</p>'
    parts.append(f"<h2>Tools</h2>{tools}")
    parts.append("<h2>Rules</h2>" + ("".join(_rule(rule) for rule in found["rules"]) or "<p>No rules.</p>"))
    if found["information"]:
        parts.append("<h2>Information</h2>" + table(
            ["name", "description", "kind", "query", "unit"],
            [[esc(item["name"]), esc(item.get("description")), esc(item.get("kind")), esc(item.get("query")),
              esc(item.get("unit"))] for item in found["information"]]))
    graph = found["graph"]
    parts.append("<h2>Graph</h2>")
    if graph["note"]:
        parts.append(f'<p class="muted">{esc(graph["note"])}</p>')
    for name in ("nodes", "edges"):
        if graph[name] is not None:
            parts.append(f"<h3>{name.capitalize()}</h3>" + table(
                graph[name]["columns"], [[esc(row.get(column)) for column in graph[name]["columns"]]
                                         for row in graph[name]["rows"]]))
    if found["collections"]:
        parts.append("<h2>Collections</h2>" + table(
            ["collection", "description", "built"],
            [[esc(item["name"]), esc(item["description"]), built_text(item["built"])]
             for item in found["collections"]]))
    return "".join(parts)


def built_text(built: dict | None) -> str:
    """When and from what a collection was built."""
    if not built:
        return '<span class="muted">not known (empty, or the server is not running)</span>'
    return esc(", ".join(f"{key}: {shown_value(value)}" for key, value in built.items()))


async def server_view(portal: Portal) -> tuple[list, dict | None, str | None]:
    """The server's tools and its describe, or the reason they are not available."""
    if not portal.server.running:
        return [], None, portal.server.error or "The portal's server is not running."
    try:
        tools = await portal.server.tools()
        described = await portal.server.call("describe", {})
    except (ToolFailed, OSError, RuntimeError) as error:
        return [], None, f"The portal's server did not answer: {error}"
    return tools, described, None


async def configuration_page(request: Request) -> Response:
    """GET /: the whole configuration, read only."""
    portal, session = request.app.state.portal, request.state.session
    tools, described, error = await server_view(portal)
    found = layer.configuration(portal.environment, tools, described, error)
    return page(portal, session, found["name"], configuration_body(found, not error))


def collections(portal: Portal) -> dict:
    """The environment's collections, from its file."""
    try:
        return layer.read_toml(portal.environment).get("collections", {})
    except ValueError:
        return {}


async def passage_page(request: Request) -> Response:
    """GET /passage: a cited passage, read through the portal's server."""
    portal, session = request.app.state.portal, request.state.session
    name, ident = request.query_params.get("collection", ""), request.query_params.get("id", "")
    label = request.query_params.get("passage") or None
    if name not in collections(portal) or not ident:
        return page(portal, session, "No such passage", "<p>No such collection or document.</p>", status=404)
    text = await read_passage(portal, name, ident, label)
    return page(portal, session, f"{ident} {label or ''}".strip(), text)


async def read_passage(portal: Portal, name: str, ident: str, label: str | None) -> str:
    """A passage as HTML, or why it cannot be shown."""
    if not portal.server.running:
        return '<p class="muted">The portal\'s server is not running, so the passage cannot be read.</p>'
    try:
        found = await portal.server.call(f"document_{name}", {"id": ident, **({"passage": label} if label else {})})
    except (ToolFailed, OSError, RuntimeError) as error:
        return f'<p class="muted">The passage cannot be read: {esc(error)}</p>'
    document = found.get("document") or {}
    if not document:
        return f'<p class="muted">{esc(name)} holds no document {esc(ident)}.</p>'
    passages = "".join(f'<p><strong>{esc(item.get("label"))}</strong> {esc(item.get("text"))}</p>'
                       for item in document.get("passages") or [])
    heading = f'<p>{esc(document.get("citation"))}: {esc(document.get("title"))}</p>'
    return heading + (passages or "<p>No such passage.</p>")


def sign_in_page(portal: Portal, message: str = "", status: int = 200) -> HTMLResponse:
    """The sign-in form."""
    shown = f'<p class="failed">{esc(message)}</p>' if message else ""
    return page(portal, None, "Sign in", f'{shown}<form method="post" action="/sign-in" class="stack">'
                '<label>Key <input type="password" name="key" autocomplete="current-password" required></label>'
                '<label>Your name, recorded with every change <input name="name" maxlength="100" required></label>'
                '<button>Sign in</button></form><p class="muted">The key is the one in CLM_PORTAL_KEY where the '
                'portal runs. The name is not checked: it is whatever you type.</p>', status=status)


async def sign_in(request: Request) -> Response:
    """GET and POST /sign-in."""
    portal = request.app.state.portal
    if request.method == "GET":
        return sign_in_page(portal)
    form = await request.form()
    key, name = str(form.get("key", "")), " ".join(str(form.get("name", "")).split())[:100]
    if not hmac.compare_digest(key.encode(), request.app.state.key.encode()):
        await asyncio.sleep(WRONG_KEY_WAIT)
        return sign_in_page(portal, "Wrong key.", 401)
    if not name:
        return sign_in_page(portal, "Give your name: it is recorded with every change you make.", 400)
    ident = secrets.token_urlsafe(32)
    session = Session(name, secrets.token_urlsafe(32), time.time() + SESSION_SECONDS)
    portal.sessions[ident] = session
    request.state.session = session
    response = await configuration_page(request)
    response.set_cookie(COOKIE, ident, max_age=SESSION_SECONDS, path="/", httponly=True, samesite="strict")
    return response


async def sign_out(request: Request) -> Response:
    """POST /sign-out."""
    request.app.state.portal.sessions.pop(request.cookies.get(COOKIE, ""), None)
    response = RedirectResponse("/sign-in", status_code=303)
    response.delete_cookie(COOKIE, path="/")
    return response


def _replay(body: bytes):
    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}
    return receive


async def _body(receive) -> bytes | None:
    chunks, size = [], 0
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            return None
        chunk = message.get("body", b"")
        size += len(chunk)
        if size > MAX_BODY:
            return None
        chunks.append(chunk)
        if not message.get("more_body"):
            return b"".join(chunks)


class Guard:  # pylint: disable=too-few-public-methods
    """Host, Origin, sign-in and CSRF checks in front of every route."""

    def __init__(self, app, portal: Portal, hosts: set[str]):
        self.app, self.portal, self.hosts = app, portal, hosts

    async def __call__(self, scope, receive, send):  # pylint: disable=too-many-return-statements
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request = Request(scope)
        host = request.headers.get("host", "")
        if host not in self.hosts:
            await PlainTextResponse("This portal answers only at its own address.", 400)(scope, receive, send)
            return
        post = scope["method"] not in ("GET", "HEAD")
        if post and request.headers.get("origin") != f"http://{host}":
            await PlainTextResponse("Refused: the request did not come from the portal's page.", 403)(
                scope, receive, send)
            return
        path = scope["path"]
        if path in OPEN_PATHS or path.startswith("/static/"):
            await self.app(scope, receive, send)
            return
        session = self.portal.session(request.cookies.get(COOKIE))
        if session is None:
            response = PlainTextResponse("Sign in first.", 403) if post else RedirectResponse("/sign-in", 303)
            await response(scope, receive, send)
            return
        if post:
            body = await _body(receive)
            if body is None:
                await PlainTextResponse("The request was too large or did not arrive.", 413)(scope, receive, send)
                return
            form = await Request(scope, _replay(body)).form()
            token = str(form.get("csrf", ""))
            if not hmac.compare_digest(token.encode(), session.csrf.encode()):
                await PlainTextResponse("Refused: the form's token is missing or wrong; reload the page.", 403)(
                    scope, receive, send)
                return
            receive = _replay(body)
        scope.setdefault("state", {})["session"] = session
        await self.app(scope, receive, send)


def allowed_hosts(host: str, port: int) -> set[str]:
    """The Host headers the portal answers: its bound address, and localhost when bound to 127.0.0.1."""
    shown = f"[{host}]" if ":" in host else host
    hosts = {f"{shown}:{port}"}
    if host == "127.0.0.1":
        hosts.add(f"localhost:{port}")
    return hosts


def routes() -> list:
    """Every route of the page."""
    return [Route("/sign-in", sign_in, methods=["GET", "POST"]), Route("/sign-out", sign_out, methods=["POST"]),
            Route("/", configuration_page), Route("/passage", passage_page),
            Mount("/static", StaticFiles(directory=STATIC), name="static")]


def create_app(portal: Portal, key: str, host: str, port: int):
    """The page, behind its checks; it starts the portal's server with it and stops it after."""
    @contextlib.asynccontextmanager
    async def lifespan(_app):
        await portal.server.start()
        try:
            yield
        finally:
            portal.server.stop()
    app = Starlette(routes=routes(), lifespan=lifespan)
    app.state.portal, app.state.key = portal, key
    return Guard(app, portal, allowed_hosts(host, port))
