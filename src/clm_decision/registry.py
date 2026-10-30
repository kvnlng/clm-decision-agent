"""The server.json that lists one deployment of the decision server in an MCP registry.

    uv run clm-registry examples/tariff/environment.toml --name io.github.you/tariff \\
        --url https://decisions.example.org/mcp --version 0.1.0 [--auth key|oauth] [--out server.json]

The server is offered as a remote server: an address an MCP client connects to over Streamable HTTP. With the
shared key, the entry says the client must send an Authorization header. With OAuth it names no header: the client
learns the token issuer from the server's own protected resource metadata. Publish the file with `mcp-publisher`,
to a private registry for a private address (docs/registry.md).
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import re
import sys
import tomllib
from pathlib import Path
from urllib.parse import urlparse

from .environment import is_environment

SCHEMA = "https://static.modelcontextprotocol.io/schemas/2025-12-11/server.schema.json"
# The registry refuses longer descriptions
DESCRIPTION_LIMIT = 100
_NAME = re.compile(r"[A-Za-z0-9.-]+/[A-Za-z0-9._-]+")
_VERSION = re.compile(r"\d+\.\d+\.\d+([-+][0-9A-Za-z.-]+)?")


def _description(path: str, given: str | None) -> str:
    if given is not None:
        description = given
    elif is_environment(path):
        description = str(tomllib.loads(Path(path).read_text(encoding="utf-8"))["environment"]["name"])
    else:
        raise ValueError("--description: a CLM scenario has none of its own; give one of at most "
                         f"{DESCRIPTION_LIMIT} characters")
    if len(description) > DESCRIPTION_LIMIT:
        raise ValueError(f"--description: at most {DESCRIPTION_LIMIT} characters, not {len(description)}")
    return description


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


# The registry's own patterns: a project's home page on GitHub or GitLab, and nothing else
_REPOSITORIES = {"github": re.compile(r"https?://(www\.)?github\.com/[\w.-]+/[\w.-]+/?"),
                 "gitlab": re.compile(r"https?://(www\.)?gitlab\.com/[\w.-]+(/[\w.-]+)+/?")}


def _source(repository: str) -> str:
    source = next((name for name, pattern in _REPOSITORIES.items() if pattern.fullmatch(repository)), None)
    # GitLab's "/-/" starts a page within a project, not the project
    if source is None or (source == "gitlab" and "/-/" in repository + "/"):
        raise ValueError("--repository: the registry takes a GitHub or GitLab project URL, such as "
                         f"https://github.com/you/repo; leave it out for any other: {repository}")
    return source


def _check(name: str, url: str, version: str, auth: str) -> None:
    if not _NAME.fullmatch(name):
        raise ValueError(f"name: must be namespace/name, such as io.github.you/tariff, not {name!r}")
    if not _VERSION.fullmatch(version):
        raise ValueError(f"--version: a version such as 1.0.0, not {version!r}")
    address = urlparse(url)
    if address.scheme != "https":
        raise ValueError(f"--url: must be https, not {url}")
    # The registry refuses localhost and bare IP addresses: an entry names a host other computers can find
    host = address.hostname or ""
    if host == "localhost" or host.endswith(".localhost") or _is_ip(host):
        raise ValueError(f"--url: name the deployment's host, not localhost or an IP address: {url}")
    # Entries are never removed from a registry, so nothing secret may ride along in the address
    if address.username or address.password or address.query or address.fragment:
        raise ValueError(f"--url: no user, password, query or fragment: the entry is published as written, "
                         f"and the key goes in the Authorization header: {url}")
    if not address.path.rstrip("/").endswith("/mcp"):
        raise ValueError(f"--url: the server answers at /mcp, so the address ends in /mcp, not {url}")
    if auth not in ("key", "oauth"):
        raise ValueError(f"--auth: key or oauth, not {auth!r}")


def server_json(path: str, name: str, url: str, version: str, auth: str, *,  # pylint: disable=too-many-arguments
                description: str | None = None, repository: str | None = None) -> dict:
    """The registry entry for the server of `path` (a scenario or an environment) at `url`; raises ValueError naming
    the setting that is wrong."""
    _check(name, url, version, auth)
    remote: dict = {"type": "streamable-http", "url": url}
    if auth == "key":
        remote["headers"] = [{"name": "Authorization", "description": "Bearer <the server's key>",
                              "isRequired": True, "isSecret": True}]
    server: dict = {"$schema": SCHEMA, "name": name, "description": _description(path, description), "version": version}
    if repository:
        server["repository"] = {"url": repository, "source": _source(repository)}
    server["remotes"] = [remote]
    return server


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description="Write the server.json that lists a deployment in an MCP registry.")
    parser.add_argument("scenario", help="the scenario file, or an environment's environment.toml, being served")
    parser.add_argument("--name", required=True, help="the registry name, such as io.github.you/tariff")
    parser.add_argument("--url", required=True, help="the address callers use, ending in /mcp")
    parser.add_argument("--version", required=True, help="this deployment's version, such as 1.0.0")
    parser.add_argument("--auth", default="key", help="key (default) or oauth, as the server is started")
    parser.add_argument("--description", help=f"at most {DESCRIPTION_LIMIT} characters (default: the environment's "
                                              "name)")
    parser.add_argument("--repository", help="the source repository's URL")
    parser.add_argument("--out", default="server.json")
    args = parser.parse_args(argv)
    try:
        server = server_json(args.scenario, args.name, args.url, args.version, args.auth,
                             description=args.description, repository=args.repository)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    Path(args.out).write_text(json.dumps(server, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}; check it with `mcp-publisher validate {args.out}`", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
