"""Answer HTTP requests for a scenario: the chat's turn, for one user, on this computer only.

    uv run python serve.py <scenario> [--port 8780]

It shows the wiring and is not a service: one conversation, kept in memory, no authentication.
docs/serving.md lists what a multi-user service would need.
"""
import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

from chat import (StartupError, load_agent, new_session, next_session, paused, reset_draft, resume_turn,
                  start_turn)
from clm_router import ClmError
from generator import GeneratorError

CHECK_KEYS = ("calls", "p", "approved", "source", "clm_approved")


def _check(check: dict) -> dict:
    """A tool check without the conversation text it was made from."""
    return {key: check[key] for key in CHECK_KEYS if key in check}


class Conversation:
    """The one conversation this process holds, and the turn waiting for a yes, if there is one."""

    def __init__(self, graph, scenario):
        self.graph, self.scenario = graph, scenario
        self.session = new_session()
        self.waiting = None       # the config that resumes a paused turn

    def handle(self, method: str, path: str, body) -> tuple[int, dict]:
        """One request -> (HTTP status, JSON answer). `body` is the parsed JSON body, or None."""
        if (method, path) == ("GET", "/"):
            return 200, self.status()
        call = {"/message": self.message, "/confirm": self.confirm, "/new": self.new}.get(path)
        if method != "POST" or call is None:
            return 404, {"error": "Not found. The calls are POST /message, POST /confirm, POST /new and GET /."}
        try:
            return call(body if isinstance(body, dict) else {})
        except (GeneratorError, ClmError) as error:
            # As in the chat, a failed turn leaves the conversation where it was
            self.waiting = None
            return 502, {"error": str(error)}

    def new(self, _body: dict) -> tuple[int, dict]:
        """Discard the request being drafted, and any call waiting for an answer; keep the conversation."""
        self.session, self.waiting = reset_draft(self.session), None
        return 200, {"status": "new"}

    def status(self) -> dict:
        """What this conversation handles and where it stands."""
        return {"scenario": self.scenario.path.name,
                "routes": {name: route.description for name, route in self.scenario.routes.items()},
                "waiting": self.waiting is not None,
                "draft_route": self.session["draft_route"], "draft": self.session["draft"]}

    def message(self, body: dict) -> tuple[int, dict]:
        """A user's message: starts a turn."""
        text = body.get("text")
        if not isinstance(text, str) or not text.strip():
            return 400, {"error": 'Send {"text": "..."} with the user\'s message.'}
        if self.waiting is not None:
            return 409, {"error": "Answer the waiting call first, with POST /confirm."}
        return self._answer(*start_turn(self.graph, self.session, text.strip()))

    def confirm(self, body: dict) -> tuple[int, dict]:
        """The user's yes or no to the tool call a turn is waiting on: resumes that turn."""
        approved = body.get("approved")
        if not isinstance(approved, bool):
            return 400, {"error": 'Send {"approved": true} or {"approved": false}.'}
        if self.waiting is None:
            return 409, {"error": "No call is waiting for an answer."}
        config = self.waiting
        return self._answer(resume_turn(self.graph, config, {"approved": approved}), config)

    def _answer(self, out: dict, config: dict) -> tuple[int, dict]:
        """A turn's output as an answer: the reply, or the call it has paused on."""
        pause = paused(out)
        if pause is not None:
            self.waiting = config
            return 200, {"status": "waiting", "check": _check(pause["check"])}
        self.waiting, self.session = None, next_session(out)
        decision = {key: value for key, value in out["decision"].items() if key != "state_text"}
        return 200, {"status": "reply", "reply": str(out["messages"][-1].content), "decision": decision,
                     "intake": out.get("intake"),
                     "tool_checks": [_check(check) for check in out.get("tool_checks", [])]}


def make_server(conversation: Conversation, port: int = 8780) -> HTTPServer:
    """An HTTP server for the conversation on this computer only; `port=0` picks a free port."""

    class Handler(BaseHTTPRequestHandler):
        """Translates HTTP to Conversation.handle and back."""

        def _respond(self, method: str) -> None:
            body = None
            raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            if raw:
                try:
                    body = json.loads(raw)
                except ValueError:
                    self._send(400, {"error": "The body is not JSON."})
                    return
            self._send(*conversation.handle(method, self.path.split("?")[0], body))

        def _send(self, status: int, answer: dict) -> None:
            data = json.dumps(answer, default=float).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # pylint: disable=invalid-name
            """Handle a GET."""
            self._respond("GET")

        def do_POST(self) -> None:  # pylint: disable=invalid-name
            """Handle a POST."""
            self._respond("POST")

        def log_message(self, format, *args) -> None:  # pylint: disable=redefined-builtin
            """Keep the terminal quiet."""

    # Not the threading server: requests are handled one at a time, which one copy of each model needs
    return HTTPServer(("127.0.0.1", port), Handler)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description="Answer HTTP requests for a scenario, for one user.")
    parser.add_argument("scenario", help="path to the scenario file")
    parser.add_argument("--port", type=int, default=8780)
    args = parser.parse_args(argv)
    try:
        scenario, graph = load_agent(args.scenario)
        server = make_server(Conversation(graph, scenario), args.port)
    except StartupError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except OSError as error:
        print(f"error: cannot listen on port {args.port}: {error}", file=sys.stderr)
        return 2
    print(f"Listening on http://127.0.0.1:{server.server_address[1]}: requests from this computer only, "
          f"no authentication, one conversation kept in memory. Ctrl-C stops it.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
