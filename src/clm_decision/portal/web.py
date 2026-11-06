"""The portal's page: sign-in, then Configuration, Ask, Review and Documents, as plain HTML from Python.

The page runs in the same process as the server agents use (live.py) and reads it directly. Every request must carry
the bound address in its Host header, which stops DNS rebinding. Every POST must come from the portal's own origin
and, once signed in, carry the session's CSRF token. Sessions are held in memory, so a restart signs everyone out.
"""
# The pages are plain HTML from Python, one module, as the rest of the portal is
# pylint: disable=too-many-lines
from __future__ import annotations

import asyncio
import contextlib
import hmac
import html
import json
import os
import re
import secrets
import shlex
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio.to_thread
import httpx
from starlette.applications import Starlette
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from . import chat, layer
from .live import NotAnswering, ToolFailed, answering
from .managed import Busy, Jobs, script, scrub

COOKIE = "clm_portal_session"
SESSION_SECONDS = 12 * 60 * 60
WRONG_KEY_WAIT = 0.5
MAX_BODY = 200 * 1024 * 1024
SIGN_IN_MAX_BODY = 16 * 1024    # a key and a name; read before the key is checked, so kept small
STATIC = Path(__file__).parent / "static"
INSPECTOR = "@modelcontextprotocol/inspector@2.13.0"
OPEN_PATHS = ("/sign-in",)
OPERATOR_NOTE = ("A change here takes effect at once: the server agents use reloads its files after every approval, "
                 "rejection and successful job, with no restart. The files are the record: commit them to git to "
                 "keep a change.")
TESTING_BANNER = ("Testing: Ask is talking to a testing server, where unapproved derived rules decide. The server "
                  "agents use is not: it admits only what its file admits.")
CHECK_SECONDS = 60      # how often the portal runs the server's checks
BACKENDS = ("openai", "claude")
STILL_ASKING = "A question is still being answered; reload Ask to see its answer, then try again."


@dataclass
class Session:
    """A signed-in person: the name they gave, the token every form carries, and their conversation, which
    `asking` guards while a question is answered."""
    name: str
    csrf: str
    expires: float
    ask: dict = field(default_factory=dict)
    asking: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False, compare=False)


class Portal:  # pylint: disable=too-few-public-methods,too-many-instance-attributes
    """What the page works on: the environment file, the server agents use (a live.LiveServer), its jobs and the
    signed-in sessions. `agents_url` is where agents reach the server, and `keys` the keys they may send (a
    server.KeySet, or None under OAuth)."""

    # pylint: disable-next=too-many-arguments
    def __init__(self, environment: Path, server, jobs: Jobs, find=script, options: dict | None = None, *,
                 agents_url: str = "http://127.0.0.1:8000/mcp", keys=None):
        self.environment, self.server, self.jobs, self.find = Path(environment), server, jobs, find
        self.options = options or {}
        self.agents_url, self.keys = agents_url, keys
        self.sessions: dict[str, Session] = {}
        self.make_model = self._model

    def started_url(self) -> str:
        """The model URL the portal started with (--model-url, or Ollama's by default)."""
        return (self.options.get("model_url") or chat.OLLAMA_V1).rstrip("/")

    def model_key(self, url: str | None) -> str | None:
        """LLM_API_KEY, only for the model URL the portal started with: a URL typed on the page never gets it."""
        return os.environ.get(chat.LLM_KEY_ENV) if (url or chat.OLLAMA_V1).rstrip("/") == self.started_url() \
            else None

    def _model(self, backend: str, model: str | None, url: str | None, http: httpx.AsyncClient):
        return make_model(backend, model, url, http, self.model_key(url))

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
    banner = f'<p class="banner testing">{TESTING_BANNER}</p>' if portal.server.testing and portal.server.running \
        else ""
    testing_error = getattr(portal.server, "testing_error", None)
    if testing_error:
        banner += f'<p class="banner failed">{esc(testing_error)}</p>'
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
    """A link to a cited passage, read through the server."""
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


def configuration_body(found: dict, agents_url: str, session: Session | None = None) -> str:
    """The Configuration page's sections, with the Reload button for a signed-in session."""
    parts = [f'<p>{esc(found["description"])}</p>']
    if found["errors"]:
        items = "".join(f"<li><pre>{esc(error)}</pre></li>" for error in found["errors"])
        parts.append(f'<section class="errors"><h2>Load errors</h2><ul>{items}</ul></section>')
    if session is not None:
        parts.append(f'<form method="post" action="/reload" class="inline">{csrf_field(session)}'
                     '<button data-confirm="Reload the server agents use from its files?">Reload</button></form>'
                     '<p class="muted">Reload reads the environment\'s files again and serves them at once; calls '
                     'already running finish with what they started with.</p>')
    parts.append('<h2>Tools, resources and prompts</h2><p>The MCP Inspector shows what the server agents use offers '
                 'and calls one tool at a time (docs/getting-started.md, "Looking inside the server"). It reaches '
                 f'the running server at {esc(agents_url)} with one of the agents\' keys:</p>'
                 f'<pre>npx -y {INSPECTOR} --cli {esc(shlex.quote(agents_url))} --transport http \\\n'
                 '    --header "Authorization: Bearer &lt;an agents\' key&gt;" --method tools/list</pre>')
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
            [[esc(item["name"]), esc(item["description"]),
              built_text(item["built"], item["left_out"], item["tools_off"])]
             for item in found["collections"]]))
    return "".join(parts)


def built_text(built: dict | None, left_out: bool = False, tools_off: bool = False) -> str:
    """When and from what a collection was built, or that the server's [serve] setting or the environment's
    [documents] tools leaves out its tools."""
    if tools_off and not built:
        return '<span class="muted">not offered: the environment\'s [documents] tools is false</span>'
    if left_out:
        return '<span class="muted">not offered: the server\'s [serve] setting leaves out its tools</span>'
    if not built:
        return '<span class="muted">not known (empty, or the server is not running)</span>'
    return esc(", ".join(f"{key}: {shown_value(value)}" for key, value in built.items()))


async def server_view(portal: Portal) -> tuple[dict | None, str | None]:
    """The server's describe, or the reason it is not available."""
    if not portal.server.running:
        return None, portal.server.error or "The server is not running."
    try:
        described = await portal.server.call("describe", {})
    except (ToolFailed, OSError, RuntimeError) as error:
        return None, f"The server did not answer: {error}"
    return described, None


async def configuration_page(request: Request, status: int = 200, message: str = "") -> Response:
    """GET /: the whole configuration, read only, with why the last reload failed, if it did, and `message` as an
    error above it."""
    portal, session = request.app.state.portal, request.state.session
    described, error = await server_view(portal)
    found = layer.configuration(portal.environment, described, error or portal.server.error)
    shown = f'<p class="failed">{esc(message)}</p>' if message else ""
    return page(portal, session, found["name"], shown + configuration_body(found, portal.agents_url, session),
                status=status)


async def reload_action(request: Request) -> Response:
    """POST /reload: reload the server agents use from its files."""
    portal = request.app.state.portal
    try:
        async with changing(portal):
            await portal.server.reload()
    except Busy as error:
        return await configuration_page(request, 409, str(error))
    if portal.server.error:
        return await configuration_page(request, 409)
    return RedirectResponse("/", status_code=303)


def collections(portal: Portal) -> dict:
    """The environment's collections, from its file."""
    try:
        return layer.read_toml(portal.environment).get("collections", {})
    except ValueError:
        return {}


async def passage_page(request: Request) -> Response:
    """GET /passage: a cited passage, read through the server."""
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
        return '<p class="muted">The server is not running, so the passage cannot be read.</p>'
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


@contextlib.asynccontextmanager
async def changing(portal: Portal):
    """Hold the lock jobs and layer writes share; Busy at once, rather than waiting, when it is held."""
    if portal.jobs.lock.locked():
        raise Busy("a job or another change is running; wait for it to finish, then try again")
    async with portal.jobs.lock:
        yield


# pylint: disable-next=too-many-arguments
def action_form(session: Session, item: dict, action: str, label: str, *, enabled: bool = True,
                extra: str = "", why: str = "") -> str:
    """One button, with a note field, posting to /review/KIND/NAME/ACTION."""
    target = f"/review/{item['kind']}/{urllib.parse.quote(item['name'], safe='')}/{action}"
    disabled = f' disabled title="{esc(why)}"' if not enabled else ""
    return (f'<form method="post" action="{esc(target)}" class="inline">{csrf_field(session)}{extra}'
            f'<input name="note" placeholder="note (optional)" maxlength="500">'
            f'<button{disabled} data-confirm="{esc(label)} {esc(item["name"])}?">{esc(label)}</button></form>')


def _decided(entry: dict) -> str:
    if not entry.get("decided_by"):
        return ""
    note = f": {esc(entry['note'])}" if entry.get("note") else ""
    return f'<p class="muted">{esc(entry.get("status"))} by {esc(entry["decided_by"])} at ' \
           f'{esc(entry.get("decided_at"))}{note}</p>'


async def _passages(portal: Portal, collection: str | None, source: dict | None) -> str:
    """Each cited passage as the server holds it, or the quote alone when it cannot be read."""
    if not source:
        return ""
    shown = source_links(collection, source)
    if collection and source.get("document") and portal.server.running:
        for label in source.get("passages") or [None]:
            shown += f'<div class="passage">{await read_passage(portal, collection, source["document"], label)}</div>'
    return shown


async def review_item(portal: Portal, session: Session, item: dict, collection: str | None) -> str:
    """One derived entry beside its passage and check, with the actions it allows."""
    entry, kind, name = item["entry"], item["kind"], item["name"]
    status = f'<span class="status {esc(item["status"])}">{esc(item["status"])}</span>'
    parts = [f'<h3>{esc(name)} <span class="origin derived">derived {esc(kind[:-1])}</span> {status}</h3>']
    if kind == "decisions":
        rule = entry.get("rule") or {}
        parts.append(f'<p>{esc(entry.get("description"))}</p><p><strong>{esc(rule.get("statement"))}</strong></p>')
        if item["supplied"]:
            parts.append(f'<p>The supplied rule of this name says: {esc(item["supplied"])}</p>')
        parts.append(dmn_table(item["table"]))
    elif kind == "facts":
        parts.append(f'<p>{esc(entry.get("node"))}.{esc(entry.get("property"))} = '
                     f'<strong>{esc(shown_value(entry.get("value")))}</strong>; supplied: '
                     f'{esc(shown_value(item["supplied"])) if item["supplied"] not in (None, "") else "none"}</p>')
    else:
        parts.append(f'<p>{esc(entry.get("target"))} {esc(entry.get("name"))}: supplied '
                     f'<strong>{esc(shown_value(entry.get("supplied")))}</strong>, the documents say '
                     f'<strong>{esc(shown_value(entry.get("derived")))}</strong>. The supplied value still decides.'
                     '</p>')
    if item["graph_row"]:
        row = item["graph_row"]
        parts.append(table(list(row), [[esc(value) for value in row.values()]]))
    if kind != "conflicts":
        parts.append(f'<p>{check_text(entry.get("check"))}</p>')
    parts.append(await _passages(portal, entry.get("collection") or collection, entry.get("source")))
    parts.append(_decided(entry))
    parts.append(review_actions(session, item, layer.has_second_reader(portal.environment)))
    return f'<section class="entry" id="{esc(kind)}-{esc(name)}">{"".join(parts)}</section>'


def review_actions(session: Session, item: dict, second_reader: bool = False) -> str:
    """The buttons an entry allows: approve (once checked and passed, or rejected after passing, to undo the
    rejection, unless a second reader must read it again first), reject, resolve, correct."""
    entry, kind = item["entry"], item["kind"]
    if kind == "conflicts":
        return action_form(session, item, "resolve", "Resolve") if entry.get("status") == "open" else ""
    passed = bool((entry.get("check") or {}).get("passed")) and entry.get("status") in ("checked", "rejected")
    why = "only an entry that passed clm-derive check can be approved"
    if passed and second_reader and entry.get("status") == "rejected":
        passed, why = False, ("a rejected entry needs a new second reading: Correct it, which sends it back to "
                              "proposed, then run clm-derive check")
    forms = [action_form(session, item, "approve", "Approve", enabled=passed, why=why),
             action_form(session, item, "reject", "Reject")]
    if kind == "facts":
        value = esc(json.dumps(entry.get("value"), ensure_ascii=False))
        forms.append(action_form(session, item, "correct", "Correct", extra=f'<label>value (JSON) '
                                 f'<input name="value" value="{value}" required></label>'))
    elif item["dmn_text"] is not None:
        forms.append('<details><summary>Correct the table</summary>' + action_form(
            session, item, "correct", "Save and check again",
            extra=f'<textarea name="dmn" required>{esc(item["dmn_text"])}</textarea>') + '</details>')
    return "".join(forms)


async def review_page(request: Request, message: str = "", status: int = 200, failed: bool = False) -> Response:
    """GET /review: derived decisions, facts and conflicts, checked first. The message is shown as an error when the
    status is one, or when `failed` says so."""
    portal, session = request.app.state.portal, request.state.session
    found = layer.review(portal.environment)
    collection = layer.read_layer(portal.environment, []).get("collection")
    parts = [f'<p class="{"failed" if failed or status >= 400 else "passed"}">{esc(message)}</p>' if message else ""]
    parts += [f'<p class="failed">{esc(error)}</p>' for error in found["errors"]]
    parts.append(job_status(portal, "/review"))
    for title, items in found["groups"]:
        parts.append(f"<h2>{esc(title)}</h2>")
        parts += [await review_item(portal, session, item, collection) for item in items]
    if not found["groups"]:
        parts.append("<p>Nothing has been derived.</p>")
    return page(portal, session, "Review", "".join(parts), status=status, changes=True)


def job_status(portal: Portal, back: str) -> str:
    """The running or last job, with its output; while one runs, the page polls and then goes back to `back`."""
    job = portal.jobs.current() or portal.jobs.last()
    if job is None:
        return ""
    if job.running:
        return (f'<p data-job-running="{esc(back)}">Running {esc(job.name)}: '
                f'<code>{esc(" ".join(job.argv))}</code> ...</p>')
    outcome = "succeeded" if job.returncode == 0 else (
        f"failed (exit code {job.returncode})" if job.returncode is not None else "did not run")
    return (f'<details class="job"><summary>Last job, {esc(job.name)}, {outcome}</summary>'
            f'<pre class="output">{esc(job.output)}</pre></details>')


ACTIONS = {"approve": ("decisions", "facts"), "reject": ("decisions", "facts"), "resolve": ("conflicts",),
           "correct": ("decisions", "facts")}
DONE = {"approve": "approved", "reject": "rejected", "resolve": "resolved"}


async def review_action(request: Request) -> Response:
    """POST /review/KIND/NAME/ACTION: approve, reject, resolve or correct one entry, as the signed-in name."""
    portal, session = request.app.state.portal, request.state.session
    kind, name, action = (request.path_params[key] for key in ("kind", "name", "action"))
    if kind not in ACTIONS.get(action, ()):
        return await review_page(request, f"{action} is not an action on {kind}", 400)
    form = await request.form()
    note = str(form.get("note", "")).strip() or None
    try:
        async with changing(portal):
            if action != "correct":
                layer.decide(portal.environment, kind, name, DONE[action], by=session.name, note=note)
                await portal.server.reload()
                return await review_page(request, f"{name} {DONE[action]}." + reloaded(portal),
                                         failed=bool(portal.server.error))
            if kind == "facts":
                try:
                    value = json.loads(str(form.get("value", "")))
                except json.JSONDecodeError as error:
                    raise layer.Refused(f"the value is not JSON: {error}") from error
                layer.correct_fact(portal.environment, name, value, by=session.name, note=note)
            else:
                layer.correct_decision(portal.environment, name, str(form.get("dmn", "")), by=session.name, note=note)
            await portal.server.reload()
    except Busy as error:
        return await review_page(request, str(error), 409)
    except layer.Refused as error:
        return await review_page(request, str(error), 409 if action == "approve" else 400)
    job = await portal.jobs.start("clm-derive check", ["clm-derive", "check", str(portal.environment)],
                                  on_success=portal.server.reload)
    return await review_page(request, f"{name} corrected: it is proposed until clm-derive check passes it"
                                      + ("" if job.running else f"; {job.output}") + ".", failed=not job.running)


def reloaded(portal: Portal) -> str:
    """Nothing when the last reload succeeded; otherwise that the server did not take the change, and why."""
    error = portal.server.error
    return f" The server agents use did not take it: {error}" if error else ""


def make_model(backend: str, model: str | None, url: str | None, http: httpx.AsyncClient, key: str | None = None):
    """The chat's model for one question: OpenAI-style (Ollama by default), sending `key` if given, or Claude, with
    the chat's defaults."""
    if backend == "claude":
        try:
            from anthropic import AsyncAnthropic  # pylint: disable=import-outside-toplevel
        except ImportError as error:
            raise chat.ClientError("Claude needs the anthropic package (clm-decision-agent[agent])") from error
        return chat.ClaudeModel(AsyncAnthropic(), model or chat.CLAUDE_MODEL)
    return chat.OpenAIModel(http, model or chat.STARTER_MODEL, url or chat.OLLAMA_V1, key)


class _OwnErrors:  # pylint: disable=too-few-public-methods
    """A model whose every failure in a turn is a ClientError, so the chat shows it as the model's error and takes
    the question back out, rather than `answering()` calling it the server not answering."""

    def __init__(self, model):
        self.model = model

    def __getattr__(self, name: str):
        return getattr(self.model, name)

    async def turn(self, messages: list[dict], tools: list[dict]):
        """The model's turn; any failure but a ClientError becomes one."""
        try:
            return await self.model.turn(messages, tools)
        except chat.ClientError:
            raise
        except Exception as error:  # pylint: disable=broad-exception-caught
            raise chat.ClientError(f"the model failed: {error or type(error).__name__}") from error


def _ask_settings(portal: Portal, form, before: tuple | None = None) -> tuple[str, str, str]:
    """The backend, model and URL a question goes to. The started --model is the default only for the started
    --backend, and a model name the page kept from the `before` settings is dropped when the backend changes."""
    started = portal.options.get("backend") or "openai"
    backend = str(form.get("backend") or started)
    if backend not in BACKENDS:
        backend = "openai"
    model = str(form.get("model") or "").strip()
    if before and before[0] != backend and model == before[1]:
        model = ""
    model = model or (portal.options.get("model") if backend == started else "") or ""
    url = str(form.get("model_url") or "").strip() or portal.options.get("model_url") or ""
    return backend, model, url


def ask_body(portal: Portal, session: Session, message: str = "", draft: str = "") -> str:
    """The conversation so far, the question form (holding `draft`) and the testing switch."""
    conversation = session.ask
    backend, model, url = conversation.get("settings") or _ask_settings(portal, {})
    shown = "".join(f'<h3>{esc(question)}</h3><div class="chat">{esc(chr(10).join(lines))}</div>'
                    for question, lines in conversation.get("turns", []))
    options = "".join(f'<option value="{name}"{" selected" if name == backend else ""}>{label}</option>'
                      for name, label in (("openai", "OpenAI-style (Ollama by default)"), ("claude", "Claude")))
    testing = portal.server.testing
    held, full = conversation.get("controls") or ("", False)
    return (f'{f"<p class=failed>{esc(message)}</p>" if message else ""}'
            '<p>Questions go to the model you choose, with the tools of the server agents use, or of the testing '
            'server while testing is on. Every call and result is shown. Tools run for real unless dry run holds them '
            'back.</p>'
            f'{shown or "<p class=muted>No questions yet.</p>"}'
            f'<form method="post" action="/ask" class="stack">{csrf_field(session)}'
            f'<label>Question <textarea name="question" required>{esc(draft)}</textarea></label>'
            f'<label>Model <select name="backend">{options}</select></label>'
            f'<label>Model name <input name="model" value="{esc(model)}" placeholder="{chat.STARTER_MODEL} or '
            f'{chat.CLAUDE_MODEL}"></label>'
            f'<label>Model URL (OpenAI-style) <input name="model_url" value="{esc(url)}" '
            f'placeholder="{chat.OLLAMA_V1}"></label>'
            f'<p class="muted">{chat.LLM_KEY_ENV}, if set, is sent only to {esc(portal.started_url())}, the model '
            'URL the portal started with; any other URL is called without it.</p>'
            '<label><input type="checkbox" name="dry_run"> Dry run: show every call, send none</label>'
            f'<label>Dry run for only these tools, comma-separated; the others run <input name="dry_run_tools" '
            f'value="{esc(held)}" placeholder="decide, decision_vesting"></label>'
            f'<label><input type="checkbox" name="full"{" checked" if full else ""}> Full results: show each result '
            f'whole, not shortened after {chat.SHOWN:,} characters</label>'
            '<button>Ask</button></form>'
            f'<form method="post" action="/ask/clear" class="inline">{csrf_field(session)}'
            '<button>New conversation</button></form>'
            f'{testing_form(session, testing) if testing or has_derived(portal) else ""}')


def testing_form(session: Session, testing: bool) -> str:
    """The switch that points Ask at a private testing server, where unapproved derived rules decide, or back at the
    server agents use. It takes effect at once, and the agents' server never changes."""
    confirm = "Point Ask back at the server agents use?" if testing else (
        "Point Ask at a private testing server where unapproved derived rules decide? The server agents use does "
        "not change.")
    return (f'<form method="post" action="/ask/testing" class="inline">{csrf_field(session)}'
            f'<input type="hidden" name="on" value="{"0" if testing else "1"}">'
            f'<button data-confirm="{esc(confirm)}">'
            f'{"Turn testing off" if testing else "Testing: let unapproved derived rules decide, in Ask only"}'
            '</button></form>')


def has_derived(portal: Portal) -> bool:
    """Whether the environment file has a [derived] table, so there are derived rules to admit."""
    try:
        return "derived" in layer.read_toml(portal.environment)
    except ValueError:
        return False


async def ask_page(request: Request) -> Response:
    """GET and POST /ask: the chat, one question at a time, kept per session. Tools named for a dry run must be the
    server's, or nothing is sent; the dry-run names and full results are kept for the next question."""
    portal, session = request.app.state.portal, request.state.session
    if request.method == "GET":
        return page(portal, session, "Ask", ask_body(portal, session))
    form = await request.form()
    question = str(form.get("question", "")).strip()
    if not question:
        return page(portal, session, "Ask", ask_body(portal, session, "Type a question."), status=400)
    if not portal.server.running:
        return page(portal, session, "Ask", ask_body(portal, session, "The server is not running: "
                                                     f"{portal.server.error or 'see Configuration'}"), status=409)
    # Refused rather than queued, as changing() does: a queued double-click would run the same tools twice
    if session.asking.locked():
        return page(portal, session, "Ask", ask_body(portal, session, STILL_ASKING, question), status=409)
    async with session.asking:
        return await _ask(portal, session, form, question)


async def _ask(portal: Portal, session: Session, form, question: str) -> Response:  # pylint: disable=too-many-locals
    """One question answered into the session's conversation, while it holds `session.asking`."""
    # With no conversation yet, the page showed the defaults, so a model name kept from them is judged the same way
    settings = _ask_settings(portal, form, session.ask.get("settings") or _ask_settings(portal, {}))
    if session.ask.get("settings") != settings:
        session.ask = {"settings": settings, "messages": None, "turns": []}
    lines: list[str] = []
    if settings[0] == "openai" and os.environ.get(chat.LLM_KEY_ENV) and portal.model_key(settings[2]) is None:
        lines.append(f"  [note: {chat.LLM_KEY_ENV} was not sent: {settings[2]} is not the model URL the portal "
                     f"started with ({portal.started_url()})]")
    named = [name.strip() for name in str(form.get("dry_run_tools", "")).split(",") if name.strip()]
    full = bool(form.get("full"))
    session.ask["controls"] = (", ".join(named), full)
    dry_run = chat.DryRun(frozenset({chat.ALL}) if form.get("dry_run") else frozenset(named))
    mark = len(session.ask["messages"] or [])
    unknown: list[str] = []
    try:
        async with httpx.AsyncClient() as http:
            model = _OwnErrors(portal.make_model(*settings, http))
            if session.ask["messages"] is None:
                session.ask["messages"] = model.start()
                mark = len(session.ask["messages"])
            async with answering(), portal.server.client() as client:
                listed = (await client.list_tools()).tools
                unknown = [] if form.get("dry_run") else sorted(set(named) - {tool.name for tool in listed})
                if not unknown:
                    await chat.answer(client, model, model.tools(listed), session.ask["messages"], question,
                                      dry_run, lines.append, full)
    except NotAnswering as error:
        if session.ask["messages"] is not None:
            del session.ask["messages"][mark:]
        return page(portal, session, "Ask", ask_body(portal, session, f"{error}; your question is kept: ask again "
                                                                       "in a moment.", question))
    except (chat.ClientError, httpx.HTTPError, OSError, RuntimeError) as error:
        lines.append(f"  [error: {error}]")
    if unknown:
        return page(portal, session, "Ask", ask_body(
            portal, session, f"Dry run names {', '.join(unknown)}, which the server does not have, so nothing was "
                             "sent: a mistyped name would let that tool run. Your question is kept.", question),
            status=400)
    session.ask["turns"].append((question, lines))
    return page(portal, session, "Ask", ask_body(portal, session))


async def ask_clear(request: Request) -> Response:
    """POST /ask/clear: start a new conversation, unless a question is still being answered."""
    portal, session = request.app.state.portal, request.state.session
    if session.asking.locked():
        return page(portal, session, "Ask", ask_body(portal, session, STILL_ASKING), status=409)
    session.ask = {}
    return RedirectResponse("/ask", status_code=303)


async def ask_testing(request: Request) -> Response:
    """POST /ask/testing: point Ask at a testing server admitting every derived status but rejected, or back."""
    portal, session = request.app.state.portal, request.state.session
    form = await request.form()
    if form.get("on") == "1" and not has_derived(portal):
        return page(portal, session, "Ask", ask_body(portal, session, f"{portal.environment.name} has no [derived] "
                                                     "table, so there are no derived rules to admit"), status=400)
    try:
        async with changing(portal):
            await portal.server.set_testing(form.get("on") == "1")
    except Busy as error:
        return page(portal, session, "Ask", ask_body(portal, session, str(error)), status=409)
    except Exception as error:  # pylint: disable=broad-exception-caught  # whatever stops the load, shown on the page
        return page(portal, session, "Ask", ask_body(portal, session, f"Testing could not start: {scrub(str(error))}"),
                    status=409)
    return RedirectResponse("/ask", status_code=303)


def jsonl_files(portal: Portal) -> list[str]:
    """The JSON-lines files in the environment's folder."""
    return sorted(path.name for path in portal.environment.parent.glob("*.jsonl") if path.is_file())


def safe_name(filename: str) -> str | None:
    """An uploaded file's name, reduced to letters, digits, dots, dashes and underscores; None unless it is a
    .jsonl file."""
    name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(filename.replace("\\", "/")).name).lstrip(".")
    return name if name.endswith(".jsonl") and len(name) > len(".jsonl") else None


def documents_body(portal: Portal, session: Session, described: dict | None, message: str = "",
                   failed: bool = False) -> str:
    """Each collection with when it was built, and the forms that load and derive."""
    found = collections(portal)
    try:
        off = layer.tools_off(layer.read_toml(portal.environment))
    except ValueError:
        off = False
    built = (described or {}).get("collections", {})
    rows = [[esc(name), esc(table_.get("description", "")),
             built_text((built.get(name) or {}).get("built"), layer.left_out(described, name), off)]
            for name, table_ in found.items()]
    choose = "".join(f'<option value="{esc(name)}">{esc(name)}</option>' for name in found)
    files = "".join(f'<option value="{esc(name)}">{esc(name)}</option>' for name in jsonl_files(portal))
    parts = [f'<p class="{"failed" if failed else "passed"}">{esc(message)}</p>' if message else "",
             job_status(portal, "/documents"), "<h2>Collections</h2>",
             table(["collection", "description", "built"], rows) if rows else "<p>No collections.</p>",
             "<h2>Load</h2><p>Replaces what the collection holds with the file, through clm-load-documents. An "
             "upload is saved in the environment's folder, so git can hold it.</p>",
             f'<form method="post" action="/documents/load" class="stack">{csrf_field(session)}'
             f'<label>Collection <select name="collection">{choose}</select></label>'
             f'<label>A file in the folder <select name="file"><option value="">(none)</option>{files}</select>'
             '</label><button data-confirm="Replace the collection with this file?">Load</button></form>',
             f'<form method="post" action="/documents/load" enctype="multipart/form-data" class="stack">'
             f'{csrf_field(session)}<label>Collection <select name="collection">{choose}</select></label>'
             '<label>Upload a .jsonl file <input type="file" name="upload" accept=".jsonl" required></label>'
             '<label><input type="checkbox" name="overwrite"> Overwrite a file of the same name</label>'
             '<button data-confirm="Save the file and replace the collection with it?">Upload and load</button>'
             '</form>', "<h2>Derive</h2>"]
    if portal.find("clm-derive") is None:
        parts.append('<p class="muted">Derivation is not available yet: clm-derive is not installed in the '
                     "portal's environment.</p>")
    else:
        parts.append('<p>Derives rules and facts from a collection into the derived layer, through clm-derive run. '
                     'Each is proposed, then checked; only a person approves one, in Review.</p>'
                     f'<form method="post" action="/documents/derive" class="stack">{csrf_field(session)}'
                     f'<label>Collection <select name="collection">{choose}</select></label>'
                     '<label>Model <select name="backend"><option value="">(clm-derive\'s default)</option>'
                     '<option value="claude">Claude</option><option value="ollama">Ollama</option></select></label>'
                     '<label>Model name <input name="model"></label>'
                     '<button data-confirm="Derive from this collection?">Derive</button></form>')
    return "".join(parts)


async def documents_page(request: Request, message: str = "", status: int = 200) -> Response:
    """GET /documents."""
    portal, session = request.app.state.portal, request.state.session
    described, _ = await server_view(portal)
    return page(portal, session, "Documents", documents_body(portal, session, described, message, status >= 400),
                status=status, changes=True)


async def _start(request: Request, name: str, argv: list[str], prepare=None) -> Response:
    """Start a job; `prepare`, such as saving an upload, runs only once the job holds the lock."""
    portal = request.app.state.portal
    try:
        job = await portal.jobs.start(name, argv, on_success=portal.server.reload, prepare=prepare)
    except Busy as error:
        return await documents_page(request, str(error), 409)
    except OSError as error:
        return await documents_page(request, f"the file cannot be saved: {error}", 500)
    if not job.running:
        return await documents_page(request, job.output, 409)
    return await documents_page(request, f"Started {name}.")


def save_aside(partial: Path, content: bytes) -> None:
    """An upload written to a hidden .partial file of its own beside where it goes, so a failed write leaves the
    file it would replace as it was."""
    with partial.open("xb") as file:
        file.write(content)


async def _load_upload(request: Request, collection: str, upload: UploadFile, overwrite: bool) -> Response:
    """Save an upload in the environment's folder and load the collection from it."""
    portal = request.app.state.portal
    name = safe_name(upload.filename)
    if name is None:
        return await documents_page(request, "only a .jsonl file can be uploaded", 400)
    folder = portal.environment.parent
    target = folder / name
    if target.exists() and not overwrite:
        return await documents_page(request, f"{name} is already in the folder; tick overwrite to replace it", 409)
    content = await upload.read()
    # Named here, so whatever ends the request, even mid-write, the finally removes it
    partial = folder / f".{name}.{secrets.token_hex(4)}.partial"
    try:
        try:    # off the event loop, so a large upload stops no other page while it is written
            await anyio.to_thread.run_sync(save_aside, partial, content)
        except OSError as error:
            return await documents_page(request, f"the file cannot be saved: {error}", 500)
        # the rename, once the job holds the lock, replaces the file in one step
        return await _start(request, f"clm-load-documents {collection}",
                            ["clm-load-documents", str(portal.environment), collection, str(target)],
                            lambda: os.replace(partial, target))
    finally:
        partial.unlink(missing_ok=True)


async def documents_load(request: Request) -> Response:
    """POST /documents/load: clm-load-documents ENVIRONMENT COLLECTION FILE, from a file in the folder or an
    upload saved there."""
    portal = request.app.state.portal
    form = await request.form()
    collection, folder = str(form.get("collection", "")), portal.environment.parent
    if collection not in collections(portal):
        return await documents_page(request, f"no collection {collection!r}", 400)
    upload = form.get("upload")
    if isinstance(upload, UploadFile) and upload.filename:
        return await _load_upload(request, collection, upload, bool(form.get("overwrite")))
    name = str(form.get("file", ""))
    if name not in jsonl_files(portal):
        return await documents_page(request, "choose a .jsonl file in the folder, or upload one", 400)
    return await _start(request, f"clm-load-documents {collection}",
                        ["clm-load-documents", str(portal.environment), collection, str(folder / name)])


async def documents_derive(request: Request) -> Response:
    """POST /documents/derive: clm-derive run ENVIRONMENT COLLECTION [--backend B] [--model M]."""
    portal = request.app.state.portal
    form = await request.form()
    collection = str(form.get("collection", ""))
    backend, model = str(form.get("backend") or ""), str(form.get("model") or "").strip()
    if collection not in collections(portal):
        return await documents_page(request, f"no collection {collection!r}", 400)
    if backend not in ("", "claude", "ollama"):
        return await documents_page(request, "the model is claude or ollama", 400)
    argv = ["clm-derive", "run", str(portal.environment), collection,
            *(["--backend", backend] if backend else []), *(["--model", model] if model else [])]
    return await _start(request, f"clm-derive run {collection}", argv)


async def jobs_current(request: Request) -> Response:
    """GET /jobs/current: whether a job is running, for the page's polling."""
    job = request.app.state.portal.jobs.current() or request.app.state.portal.jobs.last()
    return JSONResponse({"running": bool(job and job.running), "name": job.name if job else None,
                         "returncode": job.returncode if job else None})


def sign_in_page(portal: Portal, message: str = "", status: int = 200) -> HTMLResponse:
    """The sign-in form."""
    shown = f'<p class="failed">{esc(message)}</p>' if message else ""
    return page(portal, None, "Sign in", f'{shown}<form method="post" action="/sign-in" class="stack">'
                '<label>Key <input type="password" name="key" autocomplete="current-password" required></label>'
                '<label>Your name, recorded with every change <input name="name" maxlength="100" required></label>'
                '<button>Sign in</button></form><p class="muted">The key is the one in CLM_PORTAL_KEY where the '
                'portal runs. The name is not checked: it is whatever you type.</p>', status=status)


async def sign_in(request: Request) -> Response:
    """GET and POST /sign-in; signing in also drops the sessions that have expired."""
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
    now = time.time()
    portal.sessions = {found: held for found, held in portal.sessions.items() if held.expires >= now}
    ident = secrets.token_urlsafe(32)
    portal.sessions[ident] = Session(name, secrets.token_urlsafe(32), now + SESSION_SECONDS)
    # A redirect, so reloading the page that follows does not post the key again
    response = RedirectResponse("/", status_code=303)
    response.set_cookie(COOKIE, ident, max_age=SESSION_SECONDS, path="/", httponly=True, samesite="strict",
                        secure=request.app.state.secure)
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


async def _body(receive, limit: int = MAX_BODY) -> bytes | None:
    chunks, size = [], 0
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            return None
        chunk = message.get("body", b"")
        size += len(chunk)
        if size > limit:
            return None
        chunks.append(chunk)
        if not message.get("more_body"):
            return b"".join(chunks)


class Guard:  # pylint: disable=too-few-public-methods
    """Host, Origin, sign-in and CSRF checks in front of every route. `public`, when given, is the origin a
    TLS-terminating proxy serves the page at, which a POST may also come from."""

    def __init__(self, app, portal: Portal, hosts: set[str], public: str | None = None):
        self.app, self.portal, self.hosts, self.public = app, portal, hosts, public

    async def __call__(self, scope, receive, send):  # pylint: disable=too-many-return-statements,too-many-locals
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request = Request(scope)
        host = request.headers.get("host", "")
        if host not in self.hosts:
            await PlainTextResponse("This portal answers only at its own address.", 400)(scope, receive, send)
            return
        post = scope["method"] not in ("GET", "HEAD")
        origin = request.headers.get("origin")
        # At the public host only the exact public origin; at the portal's own address, its own or the public one
        own = self.public if self.public and host == urllib.parse.urlsplit(self.public).netloc else f"http://{host}"
        if post and origin != own and (self.public is None or origin != self.public):
            await PlainTextResponse("Refused: the request did not come from the portal's page.", 403)(
                scope, receive, send)
            return
        path = scope["path"]
        if path in OPEN_PATHS or path.startswith("/static/"):
            if post:
                body = await _body(receive, SIGN_IN_MAX_BODY)
                if body is None:
                    await PlainTextResponse("The request was too large or did not arrive.", 413)(scope, receive, send)
                    return
                receive = _replay(body)
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
            try:
                form = await Request(scope, _replay(body)).form()
            except MultiPartException as error:     # Starlette answers this itself only inside the app
                await PlainTextResponse(f"The form could not be read: {error.message}", 400)(scope, receive, send)
                return
            token = str(form.get("csrf", ""))
            if not hmac.compare_digest(token.encode(), session.csrf.encode()):
                await PlainTextResponse("Refused: the form's token is missing or wrong; reload the page.", 403)(
                    scope, receive, send)
                return
            receive = _replay(body)
        scope.setdefault("state", {})["session"] = session
        await self.app(scope, receive, send)


def allowed_hosts(host: str, port: int, public: str | None = None) -> set[str]:
    """The Host headers the portal answers: its bound address, localhost when bound to 127.0.0.1, and the public
    origin's host, for a proxy that passes the browser's Host on."""
    shown = f"[{host}]" if ":" in host else host
    hosts = {f"{shown}:{port}"}
    if host == "127.0.0.1":
        hosts.add(f"localhost:{port}")
    if public:
        hosts.add(urllib.parse.urlsplit(public).netloc)
    return hosts


def public_origin(text: str) -> str:
    """An origin a TLS-terminating proxy serves the page at, as a browser sends it in Origin: lower case, without a
    default port or a trailing /. ValueError, saying what is wrong, when it is not scheme://host[:port]."""
    try:
        found = urllib.parse.urlsplit(text.strip())
        port = found.port
    except ValueError as error:
        raise ValueError(f"--public-origin {text}: {error}") from error
    extra = found.path not in ("", "/") or any((found.query, found.fragment, found.username, found.password))
    if found.scheme.lower() not in ("http", "https") or not found.hostname or extra:
        raise ValueError(f"--public-origin {text}: give only the scheme, host and port the browser uses, such as "
                         "https://portal.example.org")
    scheme, host = found.scheme.lower(), found.hostname.lower()
    host = f"[{host}]" if ":" in host else host
    default = {"http": 80, "https": 443}[scheme]
    return f"{scheme}://{host}" + (f":{port}" if port not in (None, default) else "")


def routes() -> list:
    """Every route of the page."""
    return [Route("/sign-in", sign_in, methods=["GET", "POST"]), Route("/sign-out", sign_out, methods=["POST"]),
            Route("/", configuration_page), Route("/reload", reload_action, methods=["POST"]),
            Route("/passage", passage_page), Route("/review", review_page),
            # A derived fact's name holds its graph node's id, which may hold "/": the name is the rest of the path
            Route("/review/{kind}/{name:path}/{action}", review_action, methods=["POST"]),
            Route("/ask", ask_page, methods=["GET", "POST"]), Route("/ask/clear", ask_clear, methods=["POST"]),
            Route("/ask/testing", ask_testing, methods=["POST"]), Route("/documents", documents_page),
            Route("/documents/load", documents_load, methods=["POST"]),
            Route("/documents/derive", documents_derive, methods=["POST"]), Route("/jobs/current", jobs_current),
            Mount("/static", StaticFiles(directory=STATIC), name="static")]


async def run_checks(checks) -> None:
    """Run the server's checks now and every CHECK_SECONDS, until cancelled; a run that fails waits for the next."""
    while True:
        try:
            await checks.run()
        except Exception:  # pylint: disable=broad-exception-caught  # each check records its own failure
            pass
        await asyncio.sleep(CHECK_SECONDS)


def create_app(portal: Portal, key: str, host: str, port: int, public: str | None = None):
    """The page, behind its checks; while it runs, so do the server's checks, every CHECK_SECONDS. `public` is the
    origin a TLS-terminating proxy serves it at, if any, as public_origin() keeps it."""
    @contextlib.asynccontextmanager
    async def lifespan(_app):
        checks = getattr(portal.server, "checks", None)
        timer = asyncio.create_task(run_checks(checks)) if checks is not None else None
        try:
            yield
        finally:
            if timer is not None:
                timer.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await timer
    app = Starlette(routes=routes(), lifespan=lifespan)
    app.state.portal, app.state.key = portal, key
    app.state.secure = bool(public and public.startswith("https://"))     # behind TLS, never send it in clear
    return Guard(app, portal, allowed_hosts(host, port, public), public)
