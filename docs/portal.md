# The portal

The portal is a local page for the person who administers a decision server, and a command-line chat with any
running MCP server. It is part of the `clm-decision-agent` package, as `clm_decision.portal`, and runs from a
checkout with [uv](https://docs.astral.sh/uv/) as `uv run clm-portal ...`:

| Command | Option | Default | |
|---|---|---|---|
| `serve ENVIRONMENT` | `--host` | `127.0.0.1` | Another address warns that it is plain HTTP; open the page at that address, not `0.0.0.0`, since any other `Host` is refused |
| | `--port` | `8400` | |
| | `--public-origin URL` | none | The browser's address behind a TLS-ending proxy, such as `https://portal.example.org`; forms and `Host` are accepted from it, and `https://` makes the cookie `Secure` |
| | `--backend`, `--model`, `--model-url` | as the chat | Ask's starting model, changeable on the page; `LLM_API_KEY` goes only to this `--model-url`, never to one typed on the page |
| `chat URL` | `--backend` | `openai` | Any OpenAI chat API (Ollama, LM Studio, vLLM, most hosted models); `claude` is the Anthropic API, usual credentials |
| | `--model NAME` | `granite4:micro` with openai, `claude-opus-5-5` with claude | Must support tool calls |
| | `--model-url URL` | Ollama, `http://127.0.0.1:11434/v1` | OpenAI-style address |
| | `--dry-run [TOOL]` | off | Print calls and send none: every tool's, or the named tool's (repeatable). An unknown name stops the chat at start-up |
| | `--full` | off | Print results whole; otherwise shortened on screen past 1,500 characters (the model gets them whole) |

| Variable | |
|---|---|
| `CLM_PORTAL_KEY` | Signs in to the page; at least 16 characters; not the key agents use |
| `DECISION_SERVER_KEY` | The chat's key or OAuth token for the server, sent as a bearer token over https or to this computer only, never printed |
| `LLM_API_KEY` | The model's key, if it needs one |

## The page

The page is never for end users, who use their own MCP client, and agents cannot reach it: it is not an MCP server,
and the decision server has no tool that changes a status. There is one role: anyone signed in may change the server.

```bash
export CLM_PORTAL_KEY=<a long random key>                              # not the key agents use
set -a; . ./.postgres-local.env; set +a                                # if the environment has documents
uv run clm-portal serve examples/handbook/environment.toml
```

Open <http://127.0.0.1:8400> and sign in with the key and your name, which is recorded with every change and not
checked.
A wrong key waits half a second. Sessions last 12 hours in memory, so a restart signs everyone out.

**Its own server.** The portal starts `clm-server ENVIRONMENT --http` on 127.0.0.1 at a free port with a key made at
each start for that process alone, never port 8000 or `DECISION_SERVER_KEY`. It needs the environment's folder, so
it runs on the server's machine. It passes no tool selection, but `clm-server` reads the file's `[serve]` table
([environment.md](environment.md#serving-settings)), so Ask sees the agents' tools only. Without a
collection's `document_` tool, a cited passage on Review gives an error; without all of a collection's tools,
Configuration and Documents say it is not offered rather than when it was built. **The agents' server is yours to
restart** after approving changes: it reads the files at start.

### What it shows and what it changes

| Page | Shows | Changes |
|---|---|---|
| Configuration | Name, description, Inspector command ([getting started](getting-started.md#looking-inside-the-server)), every rule (supplied or derived: origin, status, owner, version, DMN rows, cited passages), information items, graph nodes and edges (Neo4j: named only), collections with build time, load errors; read from files, not the server | Nothing |
| Ask | The chat in a page: the same models, every call and result; a dry run of all or named tools (an unknown name sends nothing); **Full results** past 1,500 characters; one conversation per session | **Testing** switches its own server to `--admit-derived proposed,checked,approved`, flagged on every page; off at start, offered only with `[derived]`, dropped if that server fails to start |
| Review | Derived decisions, facts and conflicts, each with its cited passage (or quote), check, supplied value and graph row | Approve (after `clm-derive check` passes), reject or resolve, with name and note recorded. **Correct** a value or DMN table (inside `derived/`; it must load): the entry goes back to proposed and the check reruns |
| Documents | Collections | Load or upload a `.jsonl` into the folder (safe name; overwrite only if ticked) via `clm-load-documents`; **Derive** runs `clm-derive run`, unavailable without `clm-derive` |

It changes only documents and the derived layer (`derived/derived.json`, through `clm_decision.derived`, and the
tables in `derived/`), and only in the files, so git stays the record; no supplied file changes. It runs one job or
change at a time; job output shows keys and passwords as `***`, and a success restarts the portal's server.

**Security:** the cookie is `HttpOnly` and `SameSite=Strict`; every form carries a per-session token; a POST from
another origin, or a request whose `Host` is not the portal's address, is refused. Off 127.0.0.1, put it behind TLS.

## The chat

For when you have no MCP client such as Claude, or would rather not use one: a model answers from a running
server's tools, and every call and result is printed, so read those, not only the answer, and judge it yourself.
Give it no real sensitive text unless the model may see it. The starter model needs [Ollama](https://ollama.com).

```bash
ollama pull granite4:micro                                            # the starter model, about 2.1 GB
export DECISION_SERVER_KEY=<a long random key>
uv run clm-server examples/tariff/environment.toml --http             # terminal 1: the server
uv run clm-portal chat http://127.0.0.1:8000/mcp     # terminal 2: the chat
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

`uv run pytest tests/portal` runs against a stand-in server and model, with no network, key or
PostgreSQL.
