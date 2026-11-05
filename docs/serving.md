# Serving decisions to agents

`clm-server` offers a scenario's decisions as MCP tools ([README](../README.md#tools)). It loads CLM once at start-up
and answers calls until stopped; loading messages go to stderr.

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
`Authorization: Bearer <key>`** (the scheme in any case); anything else gets `401 {"error": "missing or wrong key"}`,
and the server will not start without the key variable. It listens on this computer only by default; for callers
elsewhere, set `--host`, name their address with `--allowed-host`, and put HTTPS in front, since the key travels in
every request.

A text CLM reads may be at most 32,000 characters, in `decide`, `decide_many`, `check_call` (with its call),
`correct` and an environment's judge tools; a longer one is refused. CLM reads only its last 2,048 tokens, and a
longer text would hold up other callers while it is read.

Leave out `--review` for real callers: any caller with the key could save corrections, which become training data.

For Claude Code: `claude mcp add --transport http decisions http://127.0.0.1:8000/mcp --header "Authorization: Bearer <the key>"`,
then `/mcp` to see the tools. The MCP Inspector connects as an outside agent would ([below](#inspecting-a-server)).

| Flag | Default | Meaning |
|---|---|---|
| `FILE` | | The scenario file, or an environment's `environment.toml` |
| `--review` | off | Also offer `correct` and `agreement`; scenarios only |
| `--http` | stdio | Serve Streamable HTTP at `/mcp` |
| `--host`, `--port` | `127.0.0.1`, `8000` | Where to listen |
| `--allowed-host HOST` | this computer | A host callers name (`decisions.example.org`), taken on any port, as is a page served over https from it; `127.0.0.1`, `localhost` and `[::1]` are still taken; repeatable |
| `--auth key\|oauth` | `key` | A shared key, or JWTs from an OIDC issuer (next section) |
| `--key-env VAR` | `DECISION_SERVER_KEY` | The variable holding the key |
| `--issuer URL` | `DECISION_OAUTH_ISSUER` | The token issuer; HTTPS except on this computer |
| `--audience NAME` | `DECISION_OAUTH_AUDIENCE`, else the resource URL | The audience tokens must name |
| `--scope NAME` | `DECISION_OAUTH_SCOPES`, space-separated | A scope or app role tokens must carry; repeatable |
| `--resource-url URL` | `http://HOST:PORT/mcp` | This server's public address; behind HTTPS, the one callers use |
| `--tools NAMES` | every tool | Offer only these, and `describe`; `NAME*` matches a prefix |
| `--exclude-tools NAMES` | none | Offer every tool but these; wins over `--tools` |
| `--read-only` | off | Run no acting tool ([below](#choosing-the-tools-and-a-read-only-server)) |
| `--trace FILE` | off | One JSON line per call, saying where the time went |
| `--trace-text` | off | Add each request's text to the trace |

### With tokens from your identity provider (OAuth)

Instead of a shared key, the server takes OAuth JWTs from any OIDC issuer (Keycloak, Entra ID and others), checked for
signature, issuer, audience, expiry and scopes. Each caller is known by name, and access is granted and removed in the
identity provider; the server never issues tokens.

```bash
uv run clm-server my_scenario.toml --http --auth oauth \
    --issuer https://login.example.org/realms/clm --audience decision-server --scope decide \
    --resource-url https://decisions.example.org/mcp
```

- **Accepted:** a JWT signed by one of the issuer's keys (found through its `/.well-known/openid-configuration` and
  kept), naming the issuer and audience, unexpired, carrying every `--scope`. Scopes are read from `scope` (Keycloak,
  RFC 9068), `scp` (Entra, a user's token) or `roles` (Entra, an application's token). **Without `--scope`, any token
  the issuer made for the audience is accepted.**
- **Issuer unreachable, or no usable keys:** every token is refused and a warning logged. The server asks again at
  most once a minute, so a mistyped `--issuer` shows in the log at the first call, not at start-up. The issuer, and
  the address its keys are fetched from, must be HTTPS except on this computer (`localhost`, `127.0.0.1`, `::1`).
- **401** for a missing or bad token, with a `WWW-Authenticate` header pointing to
  `/.well-known/oauth-protected-resource/mcp`, which names the issuer so an MCP client can find where to get a token;
  **403** for a token without the scope.
- **The trace names the caller:** the token's client (`client_id`, `azp` or `appid`) and subject, never the token. A
  user's subject identifies a person, so treat a shared trace as personal data.
- Not supported yet: opaque tokens (introspection), a scope per tool (one scope opens every tool), and the server as
  its own authorisation server.

**Keycloak** (checked with Keycloak 26.8.1 on a Mac, with a client credentials token):

1. A realm `clm`, and a client scope `decide` with "Include in token scope" on.
2. A confidential client for the agent, `databricks-agent`, with service accounts on and `decide` as an optional
   client scope.
3. An audience mapper on that client, adding `decision-server` to the access token's audience.
4. The issuer is `http://localhost:8081/realms/clm`. An agent gets a token from the realm's
   `/protocol/openid-connect/token` with `grant_type=client_credentials` and `scope=decide`.

**Entra ID** (not tried yet):

1. An app registration for the server: under "Expose an API", its application ID URI (`api://<server app id>`) and
   an app role `decide` (allowed member types: applications) for agents calling as themselves, as a Databricks
   connection would. Users' own tokens would need a delegated scope; Entra may refuse one with the same value as an
   app role, so that is left out here.
2. In its manifest, access token version 2: `api.requestedAccessTokenVersion` in the Microsoft Graph manifest,
   `accessTokenAcceptedVersion` in the older one. The issuer is then
   `https://login.microsoftonline.com/<tenant id>/v2.0`, as discovery gives, and the audience the server app's client
   ID. Version 1 tokens name the issuer `https://sts.windows.net/<tenant id>/` and the audience
   `api://<server app id>`, and the server refuses them all.
3. An app registration for the agent, with a client secret and the application permission `decide`, granted by an
   administrator. It asks for a token with `scope=api://<server app id>/.default`.
4. `--issuer https://login.microsoftonline.com/<tenant id>/v2.0 --audience <server app id> --scope decide`.

### Inspecting a server

The Inspector's page, and what it shows for each example, is in
[getting-started.md](getting-started.md#looking-inside-the-server). A server already running over HTTP is reached with
its address and the `Authorization` header: on the page, or from the command line (Node; `npx` downloads it), for a
check you can repeat. With the handbook server running, its documents in PostgreSQL
([examples/handbook](../examples/handbook/README.md)):

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

- `--tools NAME,NAME` offers only those, and `describe`; `--exclude-tools NAME,NAME` removes those, and wins. A name
  ending in `*` matches every tool starting with the rest (`query_*`, `search_*`, `scenario_*`); `describe` cannot be
  excluded. A name or pattern that matches no tool stops start-up with one line naming it and the tools there are.
  For a scenario, `correct` and `agreement` count as its tools with or without `--review`, so a file can exclude them.
- This limits what agents are offered; it is not an access boundary. The key or OAuth is: give a caller you would
  not trust with a tool no access to the server.
- What is not offered is gone everywhere: from the tool list; from `describe`, which lists it under `left_out` so an
  agent does not look for it; from the resources that stand for the same thing (rule resources with the `rule` tool,
  a collection's documents with `document_COLLECTION`); and from the `answer_question` prompt and the server's
  instructions where they name a tool. A scenario of an environment still runs every query, decision and
  calculation it needs: the selection limits what the agent can call, not what runs inside a scenario.
- `--read-only`: a routing scenario's `decide` and `decide_many` run every informing tool and no acting tool; each
  acting tool is reported `skipped` with the reason "read-only server", and both are marked read-only. `--review`
  writes the record, so the two together stop start-up. An environment's tools only read already, so `--read-only`
  changes nothing but `describe`. Either way `describe` says `"read_only": true` and each trace line carries
  `read_only`.
- The same settings can live in the scenario or environment file as a `[serve]` table
  ([environment.md](environment.md#serving-settings)). Each flag given replaces the file's setting, never adds to it;
  `--read-only` can only turn read-only on.

To check what a selection leaves:

```bash
npx -y @modelcontextprotocol/inspector@2.13.0 --cli uv run clm-server examples/service_desk/scenario.toml \
    --tools "decide*" --exclude-tools decide_many --method tools/list           # decide and describe
npx -y @modelcontextprotocol/inspector@2.13.0 --cli uv run clm-server examples/service_desk/scenario.toml \
    --read-only --method tools/call --tool-name decide --tool-arg "text=I need access to the readmissions data"
```

The second shows `create_ticket` skipped with the reason "read-only server", and the mock tracker is not called.

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

Each trace line gives a call's time and how much went on waiting for CLM, encoding, the heads, each of the scenario's
tools and matching stored corrections, with the texts and tokens encoded and how many came from the cache. An
environment's line counts the queries, decisions, calculations and judgements with each kind's time; a scenario's
gives the branches kept, rejected and failed, and the best. A resource read is traced the same way, named by its URI
(`rule://vesting`), with the caller and whether it failed.

The report gives each kind of call (`decide`, `check_call`, ...) a table of stages (mean, median, 95th and 99th
percentile in milliseconds, share of the time), then the cache hit rate, texts and tokens per call, padding's share
of what was encoded and, for an environment, steps per call. A stage no call had is left out.

- `--trace-text` adds the requests' text: for a scenario, the text of `decide`, `decide_many`, `check_call` and
  `correct`; for an environment, each tool call's arguments as JSON (a resource read is traced by its URI only). The
  trace then needs the same care as the corrections record ([data-handling.md](data-handling.md)); the file is
  readable by its owner only, like the record.
- Tracing adds little: a few clock readings and counts per call. A background thread writes the file; if it cannot,
  the server says so once and stops tracing, and decisions carry on.
- The MLX build applies the heads inside its own encoder, so on a Mac their time counts as encoding, and there are
  no batch or padding counts.
- A request seen before is found in CLM's cache, so repeated requests show a higher hit rate than new ones would.

### One at a time or many at once

Decisions a second (encoding's share of the time) on the example scenario's 318 example messages, each run in a fresh
server, measured before packing (below) and not measured again:

| Calls | M5 Max, 64 GB, MLX | A100 80 GB, torch |
|---|---|---|
| `decide`, one message at a time | 14.7 (95%) | 19.8 (94%) |
| `decide_many`, 100 messages a call | 22.4 (93%) | 71.6 (87%) |

On torch, a call's texts are **packed end to end, with no padding**, into rows of up to `batch_tokens` (2,048) tokens.
An attention mask keeps each text to its own tokens, and positions count as if it were alone, so every vector is the
one it would get alone. Rows go to the GPU without waiting; the vectors come back together at the end. **Each message
is encoded once for all its questions:** CLM reads a message with each question appended, so within a row the shared
start comes once and each question's tokens attend to it. Before packing, that halved the tokens. The MLX build has
its own encoder and does neither.

| Measured | Result |
|---|---|
| Row size, A100 | 1,024 and 2,048 tokens equally fast; 4,096 7% slower; 8,192 25% slower: the mask makes attention cover the whole row |
| Tools' share, A100 | The example's tools at first made a new HTTP client per call, about 17 ms each, which held `decide_many` to 23 a second until they shared one ([scenario.md](scenario.md#tools)); their share grows as CLM gets faster |
| Packing's effect on routes | A text's numbers differ very slightly encoded together, so a near tie can flip: 2 of the 318 did on the Mac, 1 on the A100, all about 0.50 against 0.49, well below the confidence threshold; no acting tool ran |
| Holding the lock | A `decide_many` call holds CLM for its batch, so other calls wait: 1,000 requests take about 20 seconds on the A100 |
| H100 80 GB (SXM) | The CrowdStrike example decided 2.5 times as fast as on the A100, 46.6 against 18.8 a second on the same 3,000 bookings, and the whole day of 99,806 in 32.5 minutes ([its README](../examples/crowdstrike/README.md#the-whole-day-on-a-data-centre-gpu)) |

## On Linux

CLM runs inside the server's process, on the backend the scenario's `[clm]` table names; which backend fits which
machine and what each needs are in [models.md](models.md), and what has been checked, the A100 included, in
[its table](models.md#what-has-been-checked). To install without a checkout, in a virtual environment:

```bash
sudo apt install build-essential python3.12-dev   # PyTorch's Triton compiles a helper on first use
pip install "clm-decision-agent[clm] @ git+https://github.com/kvnlng/clm-decision-agent"
clm-server <scenario> --http --host 0.0.0.0       # with DECISION_SERVER_KEY set
```

In a checkout, `uv sync` and `CLM_TEST_BACKEND=torch uv run pytest -m slow -k real_server -s` run CLM on the GPU
through the server.
