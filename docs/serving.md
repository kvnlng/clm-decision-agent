# Serving decisions to agents

`clm-server` offers a scenario's decisions as MCP tools ([README](../README.md#tools)). It loads CLM once at start-up
and answers calls until stopped; loading messages go to stderr. The flags, the identity-provider steps, the exact
rules of tool selection and the trace format and measurements are in [serving-reference.md](serving-reference.md).

## On this computer, over stdio

```bash
uv run clm-server my_scenario.toml [--review]
```

The client starts the server and talks over stdin and stdout. For Claude Code:

```bash
claude mcp add decisions -- uv run --directory /path/to/this/repo clm-server my_scenario.toml
```

Loading CLM takes a few seconds; if a client gives up, raise its start-up timeout (Claude Code:
`MCP_TIMEOUT=60000 claude`).

## For agents elsewhere, over HTTP

```bash
export DECISION_SERVER_KEY=<a long random key>
uv run clm-server my_scenario.toml --http [--host 127.0.0.1] [--port 8000] [--review]
```

The address is `http://<host>:<port>/mcp`, MCP's Streamable HTTP, without sessions. **Every request must send
`Authorization: Bearer <key>`**, with `Bearer` in any letter case. Anything else gets
`401 {"error": "missing or wrong key"}`, and the server will not start without the key variable. It listens on this
computer only by default; for callers elsewhere, set `--host`, name their address with `--allowed-host`, and put HTTPS
in front, since the key travels in every request.

A text CLM reads may be at most 32,000 characters, in `decide`, `decide_many`, `check_call` (with its call),
`correct` and an environment's judge tools; a longer one is refused. CLM reads only its last 2,048 tokens, and a
longer text would hold up other callers while it is read.

Leave out `--review` for real callers: any caller with the key could save corrections, which become training data.

For Claude Code: `claude mcp add --transport http decisions http://127.0.0.1:8000/mcp --header "Authorization: Bearer
<the key>"`, then `/mcp` to see the tools. The MCP Inspector connects as an outside agent would
([below](#inspecting-a-server)).

Every flag, with its default: [the flags](serving-reference.md#the-flags).

### With tokens from your identity provider (OAuth)

Instead of a shared key, the server takes OAuth JWTs from any OIDC issuer (Keycloak, Entra ID and others), checked for
signature, issuer, audience, expiry and scopes. Each caller is known by name, and access is granted and removed in the
identity provider; the server never issues tokens.

```bash
uv run clm-server my_scenario.toml --http --auth oauth \
    --issuer https://login.example.org/realms/clm --audience decision-server --scope decide \
    --resource-url https://decisions.example.org/mcp
```

What is accepted and refused, and the Keycloak and Entra ID steps:
[reference](serving-reference.md#with-tokens-from-your-identity-provider-oauth).

### Inspecting a server

The Inspector's page, and what it shows for each example, is in
[getting-started.md](getting-started.md#looking-inside-the-server). A server already running over HTTP is reached with
its address and the `Authorization` header: on the page, or from the command line, for a check you can repeat (the
command line needs Node; `npx` downloads the Inspector). With the handbook server running and its documents in
PostgreSQL ([examples/handbook](../examples/handbook/README.md)):

```bash
export DECISION_SERVER_KEY=<a long random key>
uv run clm-server examples/handbook/environment.toml --http --port 8000
# in another shell
npx -y @modelcontextprotocol/inspector@2.13.0 --cli http://127.0.0.1:8000/mcp --transport http \
    --header "Authorization: Bearer <the key>" --method resources/list
```

The other methods are `tools/list`, `resources/templates/list` and `prompts/list`. To read or get one, add
`--method resources/read --uri "rule://match_eligible"`, `--uri "document://handbook/hb-401k-2026/(2)"`, or
`--method prompts/get --prompt-name answer_question --prompt-args "question=Am I eligible for the 401(k) match?"`.
Each prints JSON: on 4 November 2026, Inspector 2.13.0 read the rule and the passage as JSON and got the prompt as
one user message ([environment.md](environment.md#resources-a-prompt-and-context)). The scenario server offers tools
only.

## Choosing the tools, and a read-only server

By default a server offers every tool it has. An operator can offer fewer, for a scenario or an environment:

```bash
# only the scenarios and reading the handbook (describe is always offered: agents start with it)
uv run clm-server examples/handbook/environment.toml --tools "describe,scenario_*,search_*,document_*"
# everything but the queries and one decision
uv run clm-server examples/tariff/environment.toml --exclude-tools "query_*,decision_news_risk"
# routing that decides and informs but opens no ticket
uv run clm-server examples/service_desk/scenario.toml --read-only
```

- This limits what agents are offered; it is not an access boundary. The key or OAuth is the boundary: a caller you
  would not trust with a tool should have no access to the server at all.
- `--read-only`: a routing scenario's `decide` and `decide_many` run every informing tool and no acting tool; each
  acting tool is reported `skipped` with the reason "read-only server", and both tools are marked read-only. `--review`
  writes the record, so the two together stop start-up. An environment's tools only read already, so `--read-only`
  changes nothing but `describe`. Either way `describe` says `"read_only": true` and each trace line carries
  `read_only`.

How names and patterns match, what a left-out tool takes with it, the file's `[serve]` table and a check with the
Inspector: [reference](serving-reference.md#choosing-the-tools-and-a-read-only-server).

## Databricks

Not tried yet. Databricks agents can call an outside MCP server over Streamable HTTP: an administrator registers it in
Unity Catalog as an MCP service, with its address and key, and Python agents, AI Playground and Genie Code can use it.
Unity Catalog's HTTP connections also offer OAuth machine-to-machine (client ID, secret, token endpoint, scope), which
would fit `--auth oauth` with an Entra app role. Databricks must reach the address: a public HTTPS address or a
private network link, allowed by your workspace's outbound network rules.

## Where the time goes

```bash
uv run clm-server my_scenario.toml --http --trace calls.jsonl
uv run clm-trace-report calls.jsonl
```

What a trace line and the report hold: [reference](serving-reference.md#where-the-time-goes). Decisions a second on a
Mac, an A100 and an H100, one at a time and in batches, and how a batch is packed:
[one at a time or many at once](serving-reference.md#one-at-a-time-or-many-at-once). The headline figures from the
CrowdStrike day's replay, 3.6 decisions a second on the Mac, 19.2 on an A100 and 51.1 on an H100, are in
[findings.md](findings.md).

## On Linux

CLM runs inside the server's process, on the backend the scenario's `[clm]` table names. Which backend fits which
machine, and what each needs, is in [models.md](models.md); what has been checked, the A100 included, is in
[its table](models.md#what-has-been-checked). To install without a checkout, in a virtual environment:

```bash
sudo apt install build-essential python3.12-dev   # PyTorch's Triton compiles a helper on first use
pip install "clm-decision-agent[clm] @ git+https://github.com/kvnlng/clm-decision-agent"
clm-server <scenario> --http --host 0.0.0.0       # with DECISION_SERVER_KEY set
```

In a checkout, `uv sync` and `CLM_TEST_BACKEND=torch uv run pytest -m slow -k real_server -s` run CLM on the GPU
through the server.
