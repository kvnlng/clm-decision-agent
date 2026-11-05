# Serving reference

The reference for [serving.md](serving.md): the server's flags, what OAuth accepts and refuses with the identity
provider steps, the exact rules of tool selection, the trace format and the measurements.

## For agents elsewhere, over HTTP

### The flags

| Flag | Default | Meaning |
|---|---|---|
| `FILE` | | The scenario file, or an environment's `environment.toml` |
| `--review` | off | Also offer `correct` and `agreement`; scenarios only |
| `--http` | stdio | Serve Streamable HTTP at `/mcp` |
| `--host`, `--port` | `127.0.0.1`, `8000` | Where to listen |
| `--allowed-host HOST` | this computer | A host that callers name (`decisions.example.org`), accepted on any port, as is a page served over https from it. `127.0.0.1`, `localhost` and `[::1]` are still accepted. Repeatable |
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

The overview has the command ([serving.md](serving.md#with-tokens-from-your-identity-provider-oauth)).

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

## Choosing the tools, and a read-only server

The overview has the commands and what `--read-only` means
([serving.md](serving.md#choosing-the-tools-and-a-read-only-server)).

- `--tools NAME,NAME` offers only those, and `describe`; `--exclude-tools NAME,NAME` removes those, and wins. A name
  ending in `*` matches every tool starting with the rest (`query_*`, `search_*`, `scenario_*`); `describe` cannot be
  excluded. A name or pattern that matches no tool stops start-up with one line naming it and the tools there are.
  For a scenario, `correct` and `agreement` count as its tools with or without `--review`, so a file can exclude them.
- What is not offered is gone everywhere: from the tool list; from `describe`, which lists it under `left_out` so an
  agent does not look for it; from the resources that stand for the same thing (rule resources with the `rule` tool,
  a collection's documents with `document_COLLECTION`); and from the `answer_question` prompt and the server's
  instructions where they name a tool. A scenario of an environment still runs every query, decision and
  calculation it needs: the selection limits what the agent can call, not what runs inside a scenario.
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

## Where the time goes

The overview has the trace and report commands ([serving.md](serving.md#where-the-time-goes)).

Each trace line gives a call's time and how much went on waiting for CLM, encoding, the heads, each of the scenario's
tools and matching stored corrections, with the texts and tokens encoded and how many came from the cache. An
environment's line counts the queries, decisions, calculations and judgements with each kind's time; a `scenario_`
tool's line gives the branches kept, rejected and failed, and the best. A resource read is traced the same way, named by
its URI (`rule://vesting`), with the caller and whether it failed.

The report gives each kind of call (`decide`, `check_call`, ...) a table of stages, with each stage's mean, median,
95th and 99th percentile in milliseconds and its share of the time. Then come the cache hit rate, texts and tokens
per call, padding's share of what was encoded and, for an environment, steps per call. A stage no call had is left out.

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

The table gives decisions a second, with encoding's share of the time in brackets, on the example scenario's 318
example messages. Each run was in a fresh server, measured before packing (below) and not measured again:

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
| Tools' share, A100 | The example's tools at first made a new HTTP client per call, about 17 ms each, which held `decide_many` to 23 a second until they shared one ([scenario reference](scenario-reference.md#tools)). Their share grows as CLM gets faster |
| Packing's effect on routes | A text's numbers differ very slightly when it is encoded with others, so a near tie can flip. 2 of the 318 did on the Mac and 1 on the A100, all about 0.50 against 0.49, well below the confidence threshold; no acting tool ran |
| Holding the lock | A `decide_many` call holds CLM for its batch, so other calls wait: 1,000 requests take about 20 seconds on the A100 |
| H100 80 GB (SXM) | The CrowdStrike example decided 2.5 times as fast as on the A100, 46.6 against 18.8 a second on the same 3,000 bookings, and the whole day of 99,806 in 32.5 minutes ([its README](../examples/crowdstrike/measurements.md#the-whole-day-on-a-data-centre-gpu)) |
