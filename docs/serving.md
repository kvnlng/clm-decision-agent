# Serving decisions to agents

`server.py` offers a scenario's decisions as MCP tools (see the table in the
[README](../README.md#the-servers-tools)). It loads CLM once at start-up, then answers calls
until it is stopped. Messages about loading go to stderr.

## On this computer, over stdio

```bash
uv run python server.py my_scenario.toml [--review]
```

The client starts the server and talks to it through its standard input and output. This is
what the test chat does, and what a desktop agent such as Claude Code does with a local server:

```bash
claude mcp add decisions -- uv run --directory /path/to/this/repo python server.py my_scenario.toml
```

Loading CLM takes a few seconds; if a client gives up waiting, raise its start-up timeout
(for Claude Code, `MCP_TIMEOUT=60000 claude`).

## For agents elsewhere, over HTTP

```bash
export DECISION_SERVER_KEY=<a long random key>
uv run python server.py my_scenario.toml --http [--host 127.0.0.1] [--port 8000] [--review]
```

- The server answers at `http://<host>:<port>/mcp`, using MCP's Streamable HTTP transport,
  without sessions.
- **Every request must send `Authorization: Bearer <key>`**; anything else gets
  `401 {"error": "missing or wrong key"}`. Without the key variable set, the server will not
  start. `--key-env` names a different variable.
- By default it listens on this computer only. To accept requests from elsewhere, set `--host`
  and name the address callers use with `--allowed-host` (for example
  `--allowed-host decisions.example.org:443`), and put HTTPS in front of it: the key travels in
  every request.
- Leave out `--review` for real callers. Review mode lets any caller with the key save
  corrections, which become training data.

Try it with the test chat, which then uses the same connection an outside agent would:

```bash
DECISION_SERVER_KEY=<the key> uv run python chat.py --url http://127.0.0.1:8000/mcp
```

Other clients:

- **Claude Code:** `claude mcp add --transport http decisions http://127.0.0.1:8000/mcp --header "Authorization: Bearer <the key>"`, then `/mcp` to see the tools.
- **The MCP Inspector**, a web page for calling a server's tools by hand:
  `npx @modelcontextprotocol/inspector` and connect to the address with the header.

## Databricks

Databricks agents can call an MCP server outside Databricks that speaks Streamable HTTP. An
administrator registers it in Unity Catalog as an MCP service, with its address and the key.
Agents written in Python, AI Playground and Genie Code can then use it. Databricks has to be
able to reach the address, so the server needs a public HTTPS address, or a private network
link, and your workspace's outbound network rules must allow it. This has not been tried yet.

## Where CLM runs

By default CLM runs inside the server's process, with MLX, on an Apple Silicon Mac. To run the
server on another machine, such as a Linux VM with a GPU, point it at the CLM authors' server
(`clm-serve`, which needs a GPU) instead: see [models.md](models.md). Two caveats:

- That backend follows the authors' server code but **has not been run against a real server**.
- Heads trained on the Mac fit only the MLX 8-bit build, so a GPU server starts with the
  released heads, and corrections are reused only for exactly the same text.
