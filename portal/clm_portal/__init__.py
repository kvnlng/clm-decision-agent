"""The portal: one local page for the person who administers a decision server, and a command-line chat.

    clm-portal chat URL [--model M] [--model-url U] [--backend openai|claude] [--dry-run [TOOL]] [--full]

`chat` talks to any running MCP server through a model, printing every call and result.
"""
from __future__ import annotations

import argparse
import sys

from . import chat


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    """The command line, with `command` naming the subcommand."""
    parser = argparse.ArgumentParser(prog="clm-portal", description="Administer a decision server from a local "
                                                                    "page, or chat with any running MCP server.")
    commands = parser.add_subparsers(dest="command", required=True)
    chat.add_arguments(commands.add_parser("chat", help="chat with a running MCP server through a model, every "
                                                        "tool call and result printed"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns the process exit code."""
    args = parse(argv)
    return chat.run(args)


if __name__ == "__main__":
    sys.exit(main())
