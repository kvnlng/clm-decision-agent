"""A stand-in for OpenMetadata's REST API, serving the service desk's dataset catalogue.

It answers the two calls openmetadata_catalogue.py makes: the list of tables (with owners, tags
and domains, page by page) and each table's lineage. A real OpenMetadata works these out by
following its graph; here the answers are written out. Any token is accepted.
"""
# Each mock stands alone, so an adopter can read one file; the HTTP plumbing repeats mock_tracker's
# pylint: disable=duplicate-code
import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

NOT_FOUND = {"code": 404, "message": "Not found"}
# name, description, owning team, tags, domain
DATASETS = [
    ("Readmissions Extract", "Admissions followed by a return to hospital within 30 days",
     "Quality Improvement", [], "Clinical Data"),
    ("Oncology Registry", "Cancer diagnoses, staging and treatment dates", "Cancer Services", ["PII.Sensitive"],
     "Clinical Data"),
    ("Pharmacy Dispensing", "Medicines dispensed by ward and date", "Pharmacy", [], "Clinical Data"),
    ("Emergency Department Attendances", "Arrivals, triage category and time to be seen", "Emergency Medicine", [],
     "Clinical Data"),
    ("Theatre Utilisation", "Scheduled and actual theatre sessions, cancellations and delays", "Surgical Services", [],
     "Clinical Data"),
    ("Outpatient Appointments", "Booked, attended and missed outpatient appointments", "Outpatient Services", [],
     "Clinical Data"),
    ("Staffing Rosters", "Planned and actual staffing by ward and shift", "Workforce", [], "Workforce"),
    ("Patient Experience Survey", "Responses to the discharge survey, by ward", "Patient Experience", [],
     "Clinical Data"),
]
# table -> the tables it is built from
UPSTREAM = {"Readmissions Extract": ["Emergency Department Attendances"]}
LINEAGE_PATH = re.compile(r"/api/v1/lineage/table/name/(.+)")


def _fqn(name: str) -> str:
    return f"warehouse.clinical.{name.lower().replace(' ', '_')}"


def table(name: str, description: str, team: str, tags: list, domain: str) -> dict:
    """One table as OpenMetadata returns it, with only the fields the reader asks for."""
    return {"id": _fqn(name), "name": name, "fullyQualifiedName": _fqn(name), "description": description,
            "owners": [{"name": team.lower().replace(" ", "-"), "displayName": team, "type": "team"}],
            "tags": [{"tagFQN": tag} for tag in tags], "domains": [{"name": domain}]}


TABLES = [table(*row) for row in DATASETS]


def tables_page(limit: int, after: str | None) -> dict:
    """A page of tables; `after` is the index of the first table on it."""
    start = int(after) if after and after.isdigit() else 0
    end = start + limit
    paging = {"total": len(TABLES), **({"after": str(end)} if end < len(TABLES) else {})}
    return {"data": TABLES[start:end], "paging": paging}


def lineage(fqn: str) -> tuple[int, dict]:
    """-> (HTTP status, body) for one table's upstream lineage."""
    found = [t for t in TABLES if t["fullyQualifiedName"] == fqn]
    if not found:
        return 404, NOT_FOUND
    sources = [t for t in TABLES if t["name"] in UPSTREAM.get(found[0]["name"], [])]
    return 200, {"entity": found[0], "nodes": sources,
                 "upstreamEdges": [{"fromEntity": s["id"], "toEntity": found[0]["id"]} for s in sources],
                 "downstreamEdges": []}


def make_server(port: int = 8585) -> ThreadingHTTPServer:
    """The HTTP server; port 0 picks a free port. `server.requests` lists the paths asked for."""

    class Handler(BaseHTTPRequestHandler):
        """Answers the two GET calls."""

        def do_GET(self) -> None:  # pylint: disable=invalid-name
            """A page of tables, or one table's lineage."""
            server.last_authorization = self.headers.get("Authorization")
            url = urlparse(self.path)
            server.requests.append(url.path)
            query = {key: values[0] for key, values in parse_qs(url.query).items()}
            match = LINEAGE_PATH.fullmatch(url.path)
            if url.path == "/api/v1/tables":
                limit = int(query["limit"]) if query.get("limit", "").isdigit() else 10
                self._send(200, tables_page(max(limit, 1), query.get("after")))
            elif match:
                self._send(*lineage(unquote(match.group(1))))
            else:
                self._send(404, NOT_FOUND)

        def _send(self, status: int, body) -> None:
            payload = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args) -> None:  # pylint: disable=redefined-builtin
            """Keep the terminal quiet."""

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.last_authorization = None
    server.requests = []
    return server


def main(argv: list[str] | None = None) -> int:
    """Run the mock OpenMetadata until interrupted."""
    parser = argparse.ArgumentParser(description="A stand-in for OpenMetadata's REST API.")
    parser.add_argument("--port", type=int, default=8585)
    args = parser.parse_args(argv)
    server = make_server(args.port)
    print(f"Mock OpenMetadata on http://127.0.0.1:{args.port} with {len(TABLES)} tables. Ctrl-C stops it.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
