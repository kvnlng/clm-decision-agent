"""The portal: one local page for the person who administers a decision server, and a command-line chat.

    clm-portal serve ENVIRONMENT.toml [--host 127.0.0.1] [--port 8400] [--backend …] [--model …] [--model-url …]
    clm-portal chat URL [--model M] [--model-url U] [--backend openai|claude] [--dry-run [TOOL]] [--full]

`serve` shows an environment's whole configuration and changes only its documents and its derived layer; sign in
with the key in CLM_PORTAL_KEY. `chat` talks to any running MCP server through a model, printing every call and
result.
"""
from __future__ import annotations

import argparse
import os
import sys

from . import chat

PORTAL_KEY_ENV = "CLM_PORTAL_KEY"
LOOPBACK = ("127.0.0.1", "localhost", "::1")


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    """The command line, with `command` naming the subcommand."""
    parser = argparse.ArgumentParser(prog="clm-portal", description="Administer a decision server from a local "
                                                                    "page, or chat with any running MCP server.")
    commands = parser.add_subparsers(dest="command", required=True)
    serve_parser = commands.add_parser("serve", help="the local page for administering an environment's server")
    serve_parser.add_argument("environment", help="the environment's environment.toml")
    serve_parser.add_argument("--host", default="127.0.0.1",
                              help="the address to listen on (default 127.0.0.1; any other is plain HTTP)")
    serve_parser.add_argument("--port", type=int, default=8400)
    serve_parser.add_argument("--backend", choices=("openai", "claude"), default="openai",
                              help="Ask's model, and Derive's: openai-style (default) or claude")
    serve_parser.add_argument("--model", help=f"default: {chat.STARTER_MODEL} with openai, {chat.CLAUDE_MODEL} with "
                                              "claude")
    serve_parser.add_argument("--model-url", help=f"the model's OpenAI-style address (default {chat.OLLAMA_V1})")
    chat.add_arguments(commands.add_parser("chat", help="chat with a running MCP server through a model, every "
                                                        "tool call and result printed"))
    return parser.parse_args(argv)


def serve(args: argparse.Namespace, run=None) -> int:
    """Serve the page until stopped; returns the process exit code. `run` serves the app (uvicorn's by default)."""
    # pylint: disable=import-outside-toplevel
    from pathlib import Path

    from clm_decision.environment import is_environment
    from clm_decision.scenario import ScenarioError

    from . import managed, web
    key = os.environ.get(PORTAL_KEY_ENV, "").strip()
    if not key:
        print(f"error: set {PORTAL_KEY_ENV} to the key that signs in to the portal; it is not the key agents use",
              file=sys.stderr)
        return 2
    environment = Path(args.environment)
    try:
        found = environment.is_file() and is_environment(environment)
    except ScenarioError:
        found = False
    if not found:
        print(f"error: {environment} is not an environment file (an environment.toml with an [environment] table)",
              file=sys.stderr)
        return 2
    if args.host not in LOOPBACK:
        print(f"warning: listening on {args.host}: the portal is plain HTTP, so its key and every page cross the "
              "network unencrypted; put it behind TLS or keep it on 127.0.0.1", file=sys.stderr)
    jobs = managed.Jobs()
    portal = web.Portal(environment, managed.ManagedServer(environment), jobs, options={
        "backend": args.backend, "model": args.model, "model_url": args.model_url})
    app = web.create_app(portal, key, args.host, args.port)
    if run is None:
        import uvicorn
        run = uvicorn.run
    print(f"Portal at http://{args.host}:{args.port}/ for {environment}; sign in with the key in {PORTAL_KEY_ENV}",
          file=sys.stderr)
    run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns the process exit code."""
    args = parse(argv)
    if args.command == "serve":
        return serve(args)
    return chat.run(args)


if __name__ == "__main__":
    sys.exit(main())
