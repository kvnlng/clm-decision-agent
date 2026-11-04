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
import os
import re
import secrets
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.datastructures import UploadFile
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from . import chat, layer
from .managed import Busy, Jobs, NotAnswering, ToolFailed, answering, script

COOKIE = "clm_portal_session"
SESSION_SECONDS = 12 * 60 * 60
WRONG_KEY_WAIT = 0.5
MAX_BODY = 200 * 1024 * 1024
STATIC = Path(__file__).parent / "static"
INSPECTOR = "@modelcontextprotocol/inspector@2.13.0"
OPEN_PATHS = ("/sign-in",)
OPERATOR_NOTE = ("The portal restarts its own server after every change. The server agents use reads the files at "
                 "start: its operator restarts it, or redeploys from git, to pick up approved changes.")
TESTING_BANNER = "Testing: unapproved derived rules are deciding."
BACKENDS = ("openai", "claude")


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
        self.make_model = make_model

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


def configuration_body(found: dict, environment: Path) -> str:
    """The Configuration page's sections."""
    parts = [f'<p>{esc(found["description"])}</p>']
    if found["errors"]:
        items = "".join(f"<li><pre>{esc(error)}</pre></li>" for error in found["errors"])
        parts.append(f'<section class="errors"><h2>Load errors</h2><ul>{items}</ul></section>')
    parts.append('<h2>Tools, resources and prompts</h2><p>The MCP Inspector shows what this server offers and calls '
                 'one tool at a time (docs/getting-started.md, "Looking inside the server"):</p>'
                 f'<pre>npx -y {INSPECTOR} uv run clm-server {esc(str(environment))}</pre>')
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


async def server_view(portal: Portal) -> tuple[dict | None, str | None]:
    """The server's describe, or the reason it is not available."""
    if not portal.server.running:
        return None, portal.server.error or "The portal's server is not running."
    try:
        described = await portal.server.call("describe", {})
    except (ToolFailed, OSError, RuntimeError) as error:
        return None, f"The portal's server did not answer: {error}"
    return described, None


async def configuration_page(request: Request) -> Response:
    """GET /: the whole configuration, read only."""
    portal, session = request.app.state.portal, request.state.session
    described, error = await server_view(portal)
    found = layer.configuration(portal.environment, described, error)
    return page(portal, session, found["name"], configuration_body(found, portal.environment))


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
    parts.append(review_actions(session, item))
    return f'<section class="entry" id="{esc(kind)}-{esc(name)}">{"".join(parts)}</section>'


def review_actions(session: Session, item: dict) -> str:
    """The buttons an entry allows: approve (once checked and passed, or rejected after passing, to undo the
    rejection), reject, resolve, correct."""
    entry, kind = item["entry"], item["kind"]
    if kind == "conflicts":
        return action_form(session, item, "resolve", "Resolve") if entry.get("status") == "open" else ""
    passed = bool((entry.get("check") or {}).get("passed")) and entry.get("status") in ("checked", "rejected")
    forms = [action_form(session, item, "approve", "Approve", enabled=passed,
                         why="only an entry that passed clm-derive check can be approved"),
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
                await portal.server.restart()
                return await review_page(request, f"{name} {DONE[action]}.")
            if kind == "facts":
                try:
                    value = json.loads(str(form.get("value", "")))
                except json.JSONDecodeError as error:
                    raise layer.Refused(f"the value is not JSON: {error}") from error
                layer.correct_fact(portal.environment, name, value, by=session.name, note=note)
            else:
                layer.correct_decision(portal.environment, name, str(form.get("dmn", "")), by=session.name, note=note)
            await portal.server.restart()
    except Busy as error:
        return await review_page(request, str(error), 409)
    except layer.Refused as error:
        return await review_page(request, str(error), 409 if action == "approve" else 400)
    job = await portal.jobs.start("clm-derive check", ["clm-derive", "check", str(portal.environment)],
                                  on_success=portal.server.restart)
    return await review_page(request, f"{name} corrected: it is proposed until clm-derive check passes it"
                                      + ("" if job.running else f"; {job.output}") + ".", failed=not job.running)


def make_model(backend: str, model: str | None, url: str | None, http: httpx.AsyncClient):
    """The chat's model for one question: OpenAI-style (Ollama by default) or Claude, with the chat's defaults."""
    if backend == "claude":
        try:
            from anthropic import AsyncAnthropic  # pylint: disable=import-outside-toplevel
        except ImportError as error:
            raise chat.ClientError("Claude needs the anthropic package (clm-decision-agent[agent])") from error
        return chat.ClaudeModel(AsyncAnthropic(), model or chat.CLAUDE_MODEL)
    return chat.OpenAIModel(http, model or chat.STARTER_MODEL, url or chat.OLLAMA_V1, os.environ.get(chat.LLM_KEY_ENV))


def _ask_settings(portal: Portal, form) -> tuple[str, str, str]:
    backend = str(form.get("backend") or portal.options.get("backend") or "openai")
    if backend not in BACKENDS:
        backend = "openai"
    model = str(form.get("model") or "").strip() or portal.options.get("model") or ""
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
    return (f'{f"<p class=failed>{esc(message)}</p>" if message else ""}'
            '<p>Questions go to the model you choose, with the tools of the portal\'s own server. Every call and '
            'result is shown. Tools run for real unless dry run is ticked.</p>'
            f'{shown or "<p class=muted>No questions yet.</p>"}'
            f'<form method="post" action="/ask" class="stack">{csrf_field(session)}'
            f'<label>Question <textarea name="question" required>{esc(draft)}</textarea></label>'
            f'<label>Model <select name="backend">{options}</select></label>'
            f'<label>Model name <input name="model" value="{esc(model)}" placeholder="{chat.STARTER_MODEL} or '
            f'{chat.CLAUDE_MODEL}"></label>'
            f'<label>Model URL (OpenAI-style) <input name="model_url" value="{esc(url)}" '
            f'placeholder="{chat.OLLAMA_V1}"></label>'
            '<label><input type="checkbox" name="dry_run"> Dry run: show every call, send none</label>'
            '<button>Ask</button></form>'
            f'<form method="post" action="/ask/clear" class="inline">{csrf_field(session)}'
            '<button>New conversation</button></form>'
            f'<form method="post" action="/ask/testing" class="inline">{csrf_field(session)}'
            f'<input type="hidden" name="on" value="{"0" if testing else "1"}">'
            f'<button data-confirm="Restart the portal\'s server{" without" if testing else " with"} unapproved '
            f'derived rules?">{"Turn testing off" if testing else "Testing: let unapproved derived rules decide"}'
            '</button></form>')


async def ask_page(request: Request) -> Response:
    """GET and POST /ask: the chat, one question at a time, kept per session."""
    portal, session = request.app.state.portal, request.state.session
    if request.method == "GET":
        return page(portal, session, "Ask", ask_body(portal, session))
    form = await request.form()
    question = str(form.get("question", "")).strip()
    if not question:
        return page(portal, session, "Ask", ask_body(portal, session, "Type a question."), status=400)
    if not portal.server.running:
        return page(portal, session, "Ask", ask_body(portal, session, "The portal's server is not running: "
                                                     f"{portal.server.error or 'see Configuration'}"), status=409)
    settings = _ask_settings(portal, form)
    if session.ask.get("settings") != settings:
        session.ask = {"settings": settings, "messages": None, "turns": []}
    lines: list[str] = []
    dry_run = chat.DryRun(frozenset({chat.ALL}) if form.get("dry_run") else frozenset())
    mark = len(session.ask["messages"] or [])
    try:
        async with httpx.AsyncClient() as http:
            model = portal.make_model(*settings, http)
            if session.ask["messages"] is None:
                session.ask["messages"] = model.start()
                mark = len(session.ask["messages"])
            async with answering(), portal.server.client() as client:
                tools = model.tools((await client.list_tools()).tools)
                await chat.answer(client, model, tools, session.ask["messages"], question, dry_run, lines.append)
    except NotAnswering as error:
        if session.ask["messages"] is not None:
            del session.ask["messages"][mark:]
        return page(portal, session, "Ask", ask_body(portal, session, f"{error}; your question is kept: ask again "
                                                                       "in a moment.", question))
    except (chat.ClientError, httpx.HTTPError, OSError, RuntimeError) as error:
        lines.append(f"  [error: {error}]")
    session.ask["turns"].append((question, lines))
    return page(portal, session, "Ask", ask_body(portal, session))


async def ask_clear(request: Request) -> Response:
    """POST /ask/clear: start a new conversation."""
    request.state.session.ask = {}
    return RedirectResponse("/ask", status_code=303)


async def ask_testing(request: Request) -> Response:
    """POST /ask/testing: restart the portal's server admitting every derived status, or not."""
    portal, session = request.app.state.portal, request.state.session
    form = await request.form()
    try:
        async with changing(portal):
            await portal.server.restart(testing=form.get("on") == "1")
    except Busy as error:
        return page(portal, session, "Ask", ask_body(portal, session, str(error)), status=409)
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
    built = (described or {}).get("collections", {})
    rows = [[esc(name), esc(table_.get("description", "")), built_text((built.get(name) or {}).get("built"))]
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
        job = await portal.jobs.start(name, argv, on_success=portal.server.restart, prepare=prepare)
    except Busy as error:
        return await documents_page(request, str(error), 409)
    except OSError as error:
        return await documents_page(request, f"the file cannot be saved: {error}", 500)
    if not job.running:
        return await documents_page(request, job.output, 409)
    return await documents_page(request, f"Started {name}.")


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
        name = safe_name(upload.filename)
        if name is None:
            return await documents_page(request, "only a .jsonl file can be uploaded", 400)
        target = folder / name
        if target.exists() and not form.get("overwrite"):
            return await documents_page(request, f"{name} is already in the folder; tick overwrite to replace it",
                                        409)
        content = await upload.read()

        def prepare() -> None:
            target.write_bytes(content)
    else:
        name = str(form.get("file", ""))
        if name not in jsonl_files(portal):
            return await documents_page(request, "choose a .jsonl file in the folder, or upload one", 400)
        target, prepare = folder / name, None
    return await _start(request, f"clm-load-documents {collection}",
                        ["clm-load-documents", str(portal.environment), collection, str(target)], prepare)


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
    """Host, Origin, sign-in and CSRF checks in front of every route. `public`, when given, is the origin a
    TLS-terminating proxy serves the page at, which a POST may also come from."""

    def __init__(self, app, portal: Portal, hosts: set[str], public: str | None = None):
        self.app, self.portal, self.hosts, self.public = app, portal, hosts, public

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
        origin = request.headers.get("origin")
        if post and origin != f"http://{host}" and (self.public is None or origin != self.public):
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
            Route("/", configuration_page), Route("/passage", passage_page), Route("/review", review_page),
            Route("/review/{kind}/{name}/{action}", review_action, methods=["POST"]),
            Route("/ask", ask_page, methods=["GET", "POST"]), Route("/ask/clear", ask_clear, methods=["POST"]),
            Route("/ask/testing", ask_testing, methods=["POST"]), Route("/documents", documents_page),
            Route("/documents/load", documents_load, methods=["POST"]),
            Route("/documents/derive", documents_derive, methods=["POST"]), Route("/jobs/current", jobs_current),
            Mount("/static", StaticFiles(directory=STATIC), name="static")]


def create_app(portal: Portal, key: str, host: str, port: int, public: str | None = None):
    """The page, behind its checks; it starts the portal's server with it and stops it after. `public` is the
    origin a TLS-terminating proxy serves it at, if any, as public_origin() keeps it."""
    @contextlib.asynccontextmanager
    async def lifespan(_app):
        await portal.server.start()
        try:
            yield
        finally:
            portal.server.stop()
    app = Starlette(routes=routes(), lifespan=lifespan)
    app.state.portal, app.state.key = portal, key
    return Guard(app, portal, allowed_hosts(host, port, public), public)
