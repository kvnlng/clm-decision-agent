# Serving decisions to agents

`clm-server` offers a scenario's decisions as MCP tools (see the table in the
[README](../README.md#tools)). It loads CLM once at start-up, then answers calls
until it is stopped. Messages about loading go to stderr.

## On this computer, over stdio

```bash
uv run clm-server my_scenario.toml [--review]
```

The client starts the server and talks to it through its standard input and output. This is
what the test chat does, and what a desktop agent such as Claude Code does with a local server:

```bash
claude mcp add decisions -- uv run --directory /path/to/this/repo clm-server my_scenario.toml
```

Loading CLM takes a few seconds; if a client gives up waiting, raise its start-up timeout
(for Claude Code, `MCP_TIMEOUT=60000 claude`).

## For agents elsewhere, over HTTP

```bash
export DECISION_SERVER_KEY=<a long random key>
uv run clm-server my_scenario.toml --http [--host 127.0.0.1] [--port 8000] [--review]
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

### With tokens from your identity provider (OAuth)

Instead of one shared key, the server can take OAuth access tokens from any OpenID Connect
issuer, such as Entra ID or Keycloak. Each caller is then known by name, and access is granted
and taken away in the identity provider. The server only checks tokens; it never issues them.

```bash
uv run clm-server my_scenario.toml --http --auth oauth \
    --issuer https://login.example.org/realms/clm --audience decision-server --scope decide \
    --resource-url https://decisions.example.org/mcp
```

- **A token is accepted** when it is a JWT signed by one of the issuer's keys (found through
  its `/.well-known/openid-configuration` and kept), names the issuer and the audience, has not
  expired, and carries every `--scope`. Scopes are read from `scope` (Keycloak, RFC 9068),
  `scp` (Entra, a user's token) or `roles` (Entra, an application's own token). **Without
  `--scope`, any token the issuer made for the audience is accepted.**
- **If the issuer cannot be reached**, every token is refused and the server logs a warning. It
  asks the issuer again at most once a minute, so a mistyped `--issuer` shows up in the log at
  the first call, not at start-up.
- **A missing or bad token gets 401**, with a `WWW-Authenticate` header that points to
  `/.well-known/oauth-protected-resource/mcp`. That document names the issuer, so an MCP client
  can find where to get a token. **A token without the scope gets 403.**
- `--audience` defaults to the resource URL, and `--resource-url` to `http://HOST:PORT/mcp`;
  behind HTTPS, set it to the address callers use. The settings can also come from
  `DECISION_OAUTH_ISSUER`, `DECISION_OAUTH_AUDIENCE` and `DECISION_OAUTH_SCOPES`
  (space-separated). The issuer must be HTTPS, except on localhost.
- **The trace names the caller:** the token's client (`client_id`, `azp` or `appid`) and
  subject, never the token. A user's subject identifies a person, so treat a trace as personal
  data when you share it.
- Not supported yet: opaque tokens (introspection), a scope per tool (one scope opens every
  tool), and the server acting as its own authorisation server.

**Keycloak** (checked with Keycloak 26.8.1 on a Mac, with a client credentials token):

1. A realm (`clm`), and a client scope `decide` with "Include in token scope" on.
2. A confidential client for the calling agent, `databricks-agent`, with service accounts on
   and `decide` as an optional client scope.
3. An audience mapper on that client, adding `decision-server` to the access token's audience.
4. The issuer is `http://localhost:8081/realms/clm`. An agent gets a token with
   `grant_type=client_credentials` and `scope=decide` from the realm's
   `/protocol/openid-connect/token`.

**Entra ID** (not tried yet):

1. An app registration for the server. Under "Expose an API", set its application ID URI
   (`api://<server app id>`). Add an app role `decide` (allowed member types: applications) for
   agents calling as themselves, as a Databricks connection would. Users' own tokens would need
   a delegated scope; Entra may refuse a scope with the same value as an app role, so that is
   left out here.
2. In its manifest, set the access token version to 2: `api.requestedAccessTokenVersion` in the
   Microsoft Graph manifest, `accessTokenAcceptedVersion` in the older one. The tokens' issuer
   is then `https://login.microsoftonline.com/<tenant id>/v2.0`, which is what discovery gives,
   and the audience is the server app's client ID. Left at version 1, tokens name the issuer
   `https://sts.windows.net/<tenant id>/` and the audience `api://<server app id>`, and the
   server refuses them all.
3. An app registration for the calling agent, with a client secret and the application
   permission `decide`, granted by an administrator. It asks for a token with
   `scope=api://<server app id>/.default`.
4. Start the server with `--issuer https://login.microsoftonline.com/<tenant id>/v2.0
   --audience <server app id> --scope decide`.

Try it with the test chat, which then uses the same connection an outside agent would:

```bash
DECISION_SERVER_KEY=<the key> uv run clm-chat --url http://127.0.0.1:8000/mcp
```

Other clients:

- **Claude Code:** `claude mcp add --transport http decisions http://127.0.0.1:8000/mcp --header "Authorization: Bearer <the key>"`, then `/mcp` to see the tools.
- **The MCP Inspector**, a web page for calling a server's tools by hand:
  `npx @modelcontextprotocol/inspector` and connect to the address with the header.

## Databricks

Databricks agents can call an MCP server outside Databricks that speaks Streamable HTTP. An
administrator registers it in Unity Catalog as an MCP service, with its address and the key.
Unity Catalog's HTTP connections also offer OAuth machine-to-machine (a client ID, a secret, a
token endpoint and a scope), which would fit `--auth oauth` with an Entra app role; that has
not been tried either.
Agents written in Python, AI Playground and Genie Code can then use it. Databricks has to be
able to reach the address, so the server needs a public HTTPS address, or a private network
link, and your workspace's outbound network rules must allow it. This has not been tried yet.

## Where the time goes

`--trace <file>` writes one line per call to the file: how long it took, and how much of that
was waiting for CLM, encoding text, the heads, each of the scenario's tools, and matching stored
corrections. It also counts the texts and tokens CLM encoded, and how many it found in its cache.
For an environment, each line counts the call's queries, decisions, calculations and judgements
with the time each kind took, and a scenario's line says how many branches were kept, rejected
and failed, and which was best.

```bash
uv run clm-server my_scenario.toml --http --trace calls.jsonl
uv run clm-trace-report calls.jsonl
```

The report gives each kind of call (`decide`, `check_call`, ...) a table of its stages, with the
mean, median, 95th and 99th percentile in milliseconds and each stage's share of the time, then
the cache hit rate, texts and tokens per call, and how much of what was encoded was padding; for
an environment, the steps per call. A stage no call had is left out.

- The trace leaves out the requests' text. `--trace-text` adds it; the trace then needs the same
  care as the corrections record ([data-handling.md](data-handling.md)).
- Tracing is off unless asked for. When on, it adds little: a few clock readings and counts per
  call. A background thread writes the file; if it cannot, the server says so once and stops
  tracing, and decisions carry on.
- On a Mac, the MLX build encodes and applies the heads inside its own code, so its trace counts
  the heads' time as encoding and has no batch or padding counts.
- A request seen before is found in CLM's cache, so repeating the same requests shows a higher
  hit rate than new ones would.

### One at a time or many at once

Measured on the example scenario's 318 example messages, each run in a fresh server, decisions a
second (encoding's share of the time):

| Calls | M5 Max, 64 GB, MLX | A100 80 GB, torch |
|---|---|---|
| `decide`, one message at a time | 14.7 (95%) | 19.8 (94%) |
| `decide_many`, 100 messages a call | 22.4 (93%) | 71.6 (87%) |

- **Texts are packed end to end, with no padding**, on torch. A call's texts go into rows of up to
  `batch_tokens` (2,048) tokens; an attention mask keeps each text to its own tokens (and a shared start, below), and
  positions count as if the text were alone, so every vector is the one the text would get alone. Rows go to the GPU one after
  another without waiting, and the vectors come back together at the end. On the A100, rows of 1,024
  and 2,048 tokens were equally fast, 4,096 was 7% slower and 8,192 25% slower: the mask makes
  attention cover the whole row.
- **Each message is encoded once for all its questions** on torch. CLM reads a message with each
  question appended, so the texts start alike; within a row, the shared start comes once and each
  question's own tokens attend to it. Before packing, that halved the tokens. The MLX build has its
  own encoder and does neither.
- Your tools' share grows as CLM gets faster. On the A100, the example's tools made a new HTTP
  client per call at first, about 17 ms each; that held `decide_many` to 23 a second until they
  shared one ([scenario.md](scenario.md#tools)).
- Encoded together, a text's numbers come out very slightly different from encoded alone. A
  decision on the edge between two routes can then go the other way: 2 of the 318 did on the
  Mac, 1 on the A100. They were near ties (about 0.50 against 0.49), well below the confidence
  threshold, so they were uncertain either way and no acting tool ran.
- A `decide_many` call holds CLM for its batch, so other calls wait: 1,000 requests take about
  20 seconds on the A100.
- On an H100 80 GB (SXM) the CrowdStrike example decided 2.5 times as fast as on the A100, 46.6
  against 18.8 a second on the same 3,000 bookings, and the whole day of 99,806 in 32.5 minutes
  ([its README](../examples/crowdstrike/README.md#the-whole-day-on-a-data-centre-gpu)). The table above
  was measured before packing and has not been measured again.

## Where CLM runs

CLM runs inside the server's process, on whatever the scenario's `[clm] backend` says
([models.md](models.md)):

- **A Mac with Apple Silicon:** `backend = "mlx"`.
- **Linux with an NVIDIA GPU of 24 GB or more**, such as a VM with an A100: `backend = "torch"`.
  A plain CPU works too, slowly.

### On Linux

Install the package with the clm extra, natively, in a virtual environment:

```bash
sudo apt install build-essential python3.12-dev   # PyTorch's Triton compiles a helper on first use
pip install "clm-decision-agent[clm] @ git+https://github.com/kvnlng/clm-decision-agent"
clm-server <scenario> --http --host 0.0.0.0       # with DECISION_SERVER_KEY set
```

- The GPU needs an NVIDIA driver of version 580 or later. Without one, CLM runs on the CPU, which
  needs about 24 GB of memory and is slow.
- Models download into Hugging Face's cache (`HF_HOME`, by default `~/.cache/huggingface`).
- In a checkout, `uv sync` and `CLM_TEST_BACKEND=torch uv run pytest -m slow -k real_server -s`
  run CLM on the GPU through the server.

What has been checked so far is listed in [models.md](models.md#what-has-been-checked), including
the A100.
