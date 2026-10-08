"""A stand-in for an issue tracker's REST API, for the service desk demo.

It answers the same requests, in the same JSON shapes, as the issue tracker of Forgejo
(https://forgejo.org) and Gitea, so the demo's tools work unchanged against a real one:

    POST /api/v1/repos/{owner}/{repo}/issues            create an issue
    GET  /api/v1/repos/{owner}/{repo}/issues/{number}   fetch one
    GET  /api/v1/repos/{owner}/{repo}/issues?q=...      search
    GET  /api/v1/repos/{owner}/{repo}/labels            list labels
    POST /api/v1/repos/{owner}/{repo}/labels            create a label

Tickets live in memory, any owner and repository name is accepted, and so are any credentials.
Run it with:
    uv run python examples/service_desk/mock_tracker.py
"""
from __future__ import annotations

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

REPO_PATH = re.compile(r"/api/v1/repos/([^/]+)/([^/]+)/(issues|labels)(?:/(\d+))?")
NOT_FOUND = {"message": "The target couldn't be found.", "errors": []}
SEED_LABELS = ["priority-high", "priority-medium", "priority-low", "governance-review"]
SEED = [
    ("Access to the readmissions extract for the quality team", "De-identified, twelve months, for the Q3 audit."),
    ("Monthly count of emergency department attendances by triage category", "Needed for the operations report."),
    ("Support for a study of theatre cancellations", "Ethics approval pending. Need help with design and data."),
]


class Tracker:
    """The tickets and labels, and the tracker's rules for creating and finding them (much simplified)."""

    def __init__(self):
        self.issues: dict[int, dict] = {}
        self.labels: dict[int, dict] = {}
        for name in SEED_LABELS:
            self.create_label({"name": name, "color": "#cccccc"})
        for title, body in SEED:
            self.create({"title": title, "body": body}, "/desk-admin/requests")

    def create_label(self, body: dict) -> tuple[int, dict]:
        """-> (HTTP status, response body), as the tracker answers a request to create a label."""
        name = str(body.get("name") or "").strip()
        if not name:
            return 422, {"message": "[Name]: Required"}
        # The real tracker accepts a second label with the same name, so callers must look before creating
        label = {"id": len(self.labels) + 1, "name": name, "color": str(body.get("color") or "#cccccc")}
        self.labels[label["id"]] = label
        return 201, label

    def create(self, body: dict, repo_url: str) -> tuple[int, dict]:
        """-> (HTTP status, response body). Like the real tracker, labels must be given as ids."""
        label_ids = body.get("labels") or []
        if not isinstance(label_ids, list) or not all(isinstance(i, int) for i in label_ids):
            return 422, {"message": "[]: json: cannot unmarshal string into CreateIssueOption.labels.0 of type int64"}
        number = len(self.issues) + 1
        self.issues[number] = {
            "number": number, "title": str(body.get("title") or ""), "body": str(body.get("body") or ""),
            "state": "open", "labels": [self.labels[i] for i in label_ids if i in self.labels],
            "html_url": f"{repo_url}/issues/{number}"}
        return 201, self.issues[number]

    def get(self, number: int) -> tuple[int, dict]:
        """-> (HTTP status, response body) for one ticket."""
        issue = self.issues.get(number)
        return (200, issue) if issue else (404, NOT_FOUND)

    def search(self, query: str, limit: int = 10) -> list[dict]:
        """Tickets whose title or body holds any word of the query; best matches first."""
        words = re.findall(r"\w+", (query or "").lower())
        scored = []
        for issue in self.issues.values():
            text = f"{issue['title']} {issue['body']}".lower()
            hits = sum(word in text for word in words)
            if hits or not words:
                scored.append((hits, issue))
        scored.sort(key=lambda pair: -pair[0])
        return [issue for _, issue in scored[:limit]]


def make_server(port: int = 8765, tracker: Tracker | None = None) -> ThreadingHTTPServer:
    """The HTTP server; port 0 picks a free port. `server.tracker` holds the tickets."""
    tracker = tracker or Tracker()

    class Handler(BaseHTTPRequestHandler):
        """Routes the requests to the tracker."""

        def _send(self, status: int, body) -> None:
            payload = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _target(self):
            self.server.last_authorization = self.headers.get("Authorization")
            url = urlparse(self.path)
            return url, REPO_PATH.fullmatch(url.path)

        def do_GET(self) -> None:  # pylint: disable=invalid-name
            """Fetch one ticket, search tickets, or list labels."""
            url, match = self._target()
            if not match:
                self._send(404, NOT_FOUND)
            elif match.group(3) == "labels":
                self._send(200, list(tracker.labels.values()))
            elif match.group(4):
                self._send(*tracker.get(int(match.group(4))))
            else:
                query = parse_qs(url.query)
                limit = query.get("limit", ["10"])[0]
                self._send(200, tracker.search(query.get("q", [""])[0], int(limit) if limit.isdigit() else 10))

        def do_POST(self) -> None:  # pylint: disable=invalid-name
            """Create a ticket or a label."""
            _, match = self._target()
            if not match or match.group(4):
                self._send(404, NOT_FOUND)
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                if not isinstance(body, dict):
                    raise ValueError
            except ValueError:
                self._send(422, {"message": "The request body must be a JSON object."})
                return
            if match.group(3) == "labels":
                self._send(*tracker.create_label(body))
            else:
                host = self.headers.get("Host", "127.0.0.1")
                self._send(*tracker.create(body, f"http://{host}/{match.group(1)}/{match.group(2)}"))

        def log_message(self, format, *args) -> None:  # pylint: disable=redefined-builtin
            """Keep the terminal quiet."""

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.tracker = tracker
    server.last_authorization = None
    return server


def main(argv: list[str] | None = None) -> int:
    """Run the mock tracker until interrupted."""
    parser = argparse.ArgumentParser(description="A stand-in for an issue tracker's REST API.")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    server = make_server(args.port)
    print(f"Mock tracker on http://127.0.0.1:{args.port} with {len(server.tracker.issues)} tickets. Ctrl-C stops it.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
