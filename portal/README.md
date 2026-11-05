# The portal

A local page for the person who administers a decision server, and a command-line chat with any running MCP
server. Not part of the `clm-decision-agent` package: it depends on it by path, and runs from a checkout with
[uv](https://docs.astral.sh/uv/) as `uv run --project portal clm-portal ...`:

| Command | Option | Default | |
|---|---|---|---|
| `serve ENVIRONMENT` | `--host` | `127.0.0.1` | Other addresses warn: plain HTTP; the address typed in the browser, not `0.0.0.0`, others refused |
| | `--port` | `8400` | |
| | `--public-origin URL` | none | Browser's address behind a TLS-ending proxy, such as `https://portal.example.org`; forms and `Host` accepted from it; `https://` makes the cookie `Secure` |
| | `--backend`, `--model`, `--model-url` | as the chat | Ask's starting model, changeable on the page; `LLM_API_KEY` only to this `--model-url`, not one typed on the page |
| `chat URL` | `--backend` | `openai` | Any OpenAI chat API (Ollama, LM Studio, vLLM, most hosted models); `claude` is the Anthropic API, usual credentials |
| | `--model NAME` | `granite4:micro` with openai, `claude-opus-5-5` with claude | Must support tool calls |
| | `--model-url URL` | Ollama, `http://127.0.0.1:11434/v1` | OpenAI-style address |
| | `--dry-run [TOOL]` | off | Print calls, send none: every tool's, or the named tool's (repeatable); an unknown name stops the chat at start-up |
| | `--full` | off | Print results whole; otherwise shortened on screen past 1,500 characters (the model gets them whole) |

| Variable | |
|---|---|
| `CLM_PORTAL_KEY` | Signs in to the page; at least 16 characters; not the key agents use |
| `DECISION_SERVER_KEY` | The chat's key or OAuth token for the server, sent as a bearer token over https or to this computer only, never printed |
| `LLM_API_KEY` | The model's key, if it needs one |

## The page

Never for end users (they use their own MCP client), and unreachable by agents: not an MCP server, and the decision
server has no tool that changes a status. One role: anyone signed in may change the server.

```bash
export CLM_PORTAL_KEY=<a long random key>                              # not the key agents use
set -a; . ./.postgres-local.env; set +a                                # if the environment has documents
uv run --project portal clm-portal serve examples/handbook/environment.toml
```

Open <http://127.0.0.1:8400> and sign in with the key and your name, recorded with every change and not checked.
A wrong key waits half a second. Sessions last 12 hours in memory, so a restart signs everyone out.

**Its own server.** The portal starts `clm-server ENVIRONMENT --http` on 127.0.0.1 at a free port with a key made at
each start for that process alone, never port 8000 or `DECISION_SERVER_KEY`. It needs the environment's folder, so
it runs on the server's machine. It passes no tool selection, but `clm-server` reads the file's `[serve]` table
([environment.md](../docs/environment.md#serving-settings)), so Ask sees the agents' tools only; without a
collection's `document_` tool a cited passage on Review errors, and without all its tools Configuration and
Documents say it is not offered, not when it was built. **The agents' server is yours to restart** after approving
changes: it reads the files at start.

### What it shows and what it changes

| Page | Shows | Changes |
|---|---|---|
| Configuration | Name, description, Inspector command ([getting started](../docs/getting-started.md#looking-inside-the-server)), every rule (supplied or derived: origin, status, owner, version, DMN rows, cited passages), information items, graph nodes and edges (Neo4j: named only), collections with build time, load errors; read from files, not the server | Nothing |
| Ask | The chat in a page: same models, every call and result; dry run, all or named tools (unknown name: nothing sent); **Full results** past 1,500 characters; conversation per session | **Testing**: own server on `--admit-derived proposed,checked,approved`, flagged on every page; off at start, only with `[derived]`, dropped if it fails to start |
| Review | Derived decisions, facts, conflicts, each with cited passage (or quote), check, supplied value, graph row | Approve (after `clm-derive check` passes), reject, resolve; name and note recorded. **Correct** a value or DMN table (inside `derived/`, must load): back to proposed, check reruns |
| Documents | Collections | Load or upload a `.jsonl` into the folder (safe name; overwrite only if ticked) via `clm-load-documents`; **Derive** runs `clm-derive run`, unavailable without `clm-derive` |

It changes only documents and the derived layer (`derived/derived.json`, through `clm_decision.derived`, and the
tables in `derived/`), in the files, so git stays the record; no supplied file changes. One job or change at a time;
job output shows keys and passwords as `***`, and success restarts the portal's server.

**Security:** cookie `HttpOnly` and `SameSite=Strict`; a per-session token in every form; a POST from another origin,
or a request whose `Host` is not the portal's address, refused. Off 127.0.0.1, put it behind TLS.

## The chat

For when you have no MCP client such as Claude, or would rather not use one: a model answers from a running
server's tools, and every call and result is printed, so read those, not only the answer, and judge it yourself.
Give it no real sensitive text unless the model may see it. The starter model needs [Ollama](https://ollama.com).

```bash
ollama pull granite4:micro                                            # the starter model, about 2.1 GB
export DECISION_SERVER_KEY=<a long random key>
uv run clm-server examples/tariff/environment.toml --http             # terminal 1: the server
uv run --project portal clm-portal chat http://127.0.0.1:8000/mcp     # terminal 2: the chat
```

```
you> Who supplies the components of Product Line X?
  [tool: query_components_of {"product": "PX"}]
  [result: ...]
Product Line X has three components: ...
```

Follow-ups keep the conversation; `/tools` lists the tools; `/quit` leaves. **Every tool call is carried out by
default**, acting tools included; point them at a test system if they must not act for real. A held-back call prints
as `[dry run, not sent: TOOL {arguments}]` and the model is told. Acting tools run inside `decide`, so holding back
`decide` skips the whole decision; a stand-in such as the service desk's `mock_tracker.py` shows decisions without
real actions.

**The starter model**, `granite4:micro`, beat `qwen3:4b` and `llama3.2:3b`: fastest, and the only one to call
`decide`. Same servers, three questions, once each:

| Question | `granite4:micro`, local, free | `claude-opus-5-5`, paid |
|---|---|---|
| Who supplies Product Line X's components? | Right | Right |
| Should sourcing move to Region B? | Passed the product's name, not its id `PX`; every branch failed | Right: 50%, and why 100% was rejected |
| Service desk: who handles a registry request? | Right route and ticket, but rewrote the ticket's address | Right, and did not claim more than the result showed |

- **A weak answer may be the model, not your server:** every miss was the small model's; Claude got all three on
  the same tools. Try a larger model before changing tools.
- **A small model shows unclear descriptions:** two passed a component's name, not its id; "give the id, such as
  C1" in the tariff descriptions fixed `granite4:micro`. Test with both sizes.
- **Speed is similar:** 4 to 7 seconds a question for `granite4:micro` on a Mac, about 10 for Claude.

## Tests

`uv run --project portal pytest portal/tests`: a stand-in server and model; no network, key or PostgreSQL.
