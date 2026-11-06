# The portal

The portal is a page for the person who administers a decision server. It is served by the server itself, on a
second port of the same process, so it sees exactly what agents see. It is part of the `clm-decision-agent` package
(`clm_decision.portal`) and runs from a checkout with [uv](https://docs.astral.sh/uv/).

```bash
export DECISION_SERVER_KEY=<a long random key>                         # the agents' key
export CLM_PORTAL_KEY=<another long random key>                        # signs in to the page
set -a; . ./.postgres-local.env; set +a                                # if the environment has documents
uv run clm-server examples/handbook/environment.toml --http --portal 8400
```

Agents use <http://127.0.0.1:8000/mcp> as before; the page is at <http://127.0.0.1:8400>. Sign in with the portal key
and your name, which is recorded with every change and not checked. A wrong key waits half a second. Sessions last 12
hours in memory, so a restart signs everyone out.

| Flag of `clm-server` | Default | |
|---|---|---|
| `--portal PORT` | off | Serve the portal on this port. Needs `--http`, an environment file (not a scenario) and `CLM_PORTAL_KEY` |
| `--portal-host HOST` | `127.0.0.1` | Another address warns that it is plain HTTP; open the page at that address, not `0.0.0.0`, since any other `Host` is refused |
| `--portal-public-origin URL` | none | The browser's address behind a TLS-ending proxy, such as `https://portal.example.org`; forms and `Host` are accepted from it, and `https://` makes the cookie `Secure` |
| `--ask-backend`, `--ask-model`, `--ask-model-url` | [as Ask](#ask) | Ask's starting model, changeable on the page; `LLM_API_KEY` goes only to this `--ask-model-url`, never to one typed on the page |

| Variable | |
|---|---|
| `CLM_PORTAL_KEY` | Signs in to the page; at least 16 characters; not a key agents use |
| `DECISION_SERVER_KEY` | The agents' key (or `--key-file` keys); Ask calls the server in the same process, so it needs none |
| `LLM_API_KEY` | The model's key, if it needs one |

## The page

The page is never for end users, who use their own MCP client, and agents cannot reach it: it is not an MCP server,
and the decision server has no tool that changes a status. There is one role: anyone signed in may change the server.

**One server.** There is no second server to restart. An approval on Review, a derive or load job that succeeds and
the Reload button read the environment's files again and serve them at once, while calls keep running; a reload that
fails leaves the old environment serving and says why. The page keeps to the file's `[serve]` table
([environment.md](environment.md#serving-settings)), as the agents do. Without a collection's `document_` tool, a
cited passage on Review gives an error; without all of a collection's tools, Configuration and Documents say it is not
offered rather than when it was built. Jobs run with neither the portal key nor the agents' key.

### What it shows and what it changes

| Page | Shows | Changes |
|---|---|---|
| Configuration | Name, description, Inspector command ([getting started](getting-started.md#looking-inside-the-server)), every rule (supplied or derived: origin, status, owner, version, DMN rows, cited passages), information items, graph nodes and edges (Neo4j: named only), collections with build time, load errors; read from files, not the server | Nothing; **Reload** reads the files again |
| Ask | A model answers from the server's tools, every call and result shown ([below](#ask)); a dry run of all or named tools (an unknown name sends nothing); **Full results** past 1,500 characters; one conversation per session | **Testing** gives Ask its own private server that admits derived entries with status `proposed`, `checked` or `approved`, flagged on every page; off at start, offered only with `[derived]`. Agents never see unapproved rules |
| Review | Derived decisions, facts and conflicts, each with its cited passage (or quote), check, supplied value and graph row | Approve (after `clm-derive check` passes), reject or resolve, with name and note recorded; an approval reloads. **Correct** a value or DMN table (inside `derived/`; it must load): the entry goes back to proposed and the check reruns |
| Documents | Collections | Load or upload a `.jsonl` into the folder (safe name; overwrite only if ticked) via `clm-load-documents`; **Derive** runs `clm-derive run`, unavailable without `clm-derive` |
| Server | The running server: name, files, versions, mode, offered tools, checks, credentials | **Run checks now**; **Disable** or **Enable** a tool |

The Server page is described next. The other pages change only documents and the derived layer
(`derived/derived.json`, through `clm_decision.derived`, and the tables in `derived/`), and only in the files, so git
stays the record; no supplied file changes. They run one job or change at a time; job output shows keys and passwords
as `***`.

### The Server page

- **Checks** run every 60 seconds and on demand, and show each one's status, detail, last success and last error, and
  the tools it affects: `clm`, `postgres`, `neo4j`, `oauth`, `trace`, `tools` and `describe`. A tool's
  `infrastructure` ([reference](environment-reference.md#the-file)) says which checks it depends on. The
  server's own calls are marked `synthetic` in the trace and left out of `clm-trace-report`.
- **Tools** lists the offered tools, with whether each is read only, enabled or disabled, and what it runs on. Disable
  stops a tool at once (a call is refused) and tells agents the list changed, but a client may keep the list it
  fetched for up to 30 seconds, longer if it is an older client. The choice is kept in `disabled_tools.json` beside
  the environment file, so it survives a restart. `describe` cannot be disabled.
- **Credentials** show each key an agent can send by its id (the first 8 hex characters of its SHA-256) and when it
  was last used, OAuth issuer, audience and scopes, and `set` or `missing` for the PostgreSQL and Neo4j passwords
  and `LLM_API_KEY`. No value is shown. To rotate an agents' key, add the new one to the `--key-file` file, reload or
  restart, move the agents over, then delete the old line ([serving-reference.md](serving-reference.md#the-flags)).
  A reload that cannot read the key file (missing, a key too short, no keys, or the portal's own key) keeps the old
  keys and says why.

**Security:** the cookie is `HttpOnly` and `SameSite=Strict`; every form carries a per-session token; a POST from
another origin, or a request whose `Host` is not the portal's address, is refused. Off 127.0.0.1, put it behind TLS.

## Ask

For a quick look without an MCP client such as Claude: a model answers from the server's own tools, and every call
and result is shown, so read those, not only the answer, and judge it yourself. Give it no real sensitive text unless
the model may see it. The starter model needs [Ollama](https://ollama.com) (`ollama pull granite4:micro`, about 2.1
GB).

| Setting | Default | |
|---|---|---|
| Backend | `openai` | Any OpenAI chat API (Ollama, LM Studio, vLLM, most hosted models); `claude` is the Anthropic API, usual credentials |
| Model | `granite4:micro` with openai, `claude-opus-5-5` with claude | Must support tool calls |
| Model URL | Ollama, `http://127.0.0.1:11434/v1` | OpenAI-style address |
| Dry run | off | Show calls and send none: every tool's, or the named tools'. An unknown name sends nothing |
| Full results | off | Show results whole; otherwise shortened past 1,500 characters (the model gets them whole) |

The `--ask-*` flags set the first three when the server starts; the page can change them. Follow-ups keep the
conversation, one per session. **Every tool call is carried out by default**, acting tools included; point them at a
test system if they must not act for real. A held-back call shows as `[dry run, not sent: TOOL {arguments}]` and the
model is told. Acting tools run inside `decide`, so holding back `decide` skips the whole decision; a stand-in such as
the service desk's `mock_tracker.py` shows decisions without real actions.

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
