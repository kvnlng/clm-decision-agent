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

CLM runs inside the server's process, on whatever the scenario's `[clm] backend` says
([models.md](models.md)):

- **A Mac with Apple Silicon:** `backend = "mlx"`.
- **Linux with an NVIDIA GPU of 24 GB or more**, such as a VM with an A100: `backend = "torch"`.
  A plain CPU works too, slowly.

### In Docker

The `Dockerfile` installs the project on Linux:

```bash
docker build -t clm-decision .
docker run --rm clm-decision                       # the fast tests
docker run --rm --gpus all -v clm-hf:/cache -e CLM_TEST_BACKEND=torch clm-decision \
    uv run pytest -m slow -k real_server -s        # CLM on the GPU, through the server
```

- Models download into `/cache`; the volume keeps them between runs.
- `--gpus all` needs the NVIDIA container toolkit, and a driver of version 580 or later on the host. Without it, CLM runs on the CPU, which
  needs about 24 GB of memory for the container and is slow.
- To serve from the container, run `uv run python server.py <scenario> --http --host 0.0.0.0`
  with the key variable set (`-e DECISION_SERVER_KEY=...`) and the port published (`-p 8000:8000`).

What has been checked so far is listed in [models.md](models.md#what-has-been-checked), including
this image on an A100.
