# The portal

One local page for the person who administers a decision server, and a command-line chat with any running MCP
server. It is not part of the `clm-decision-agent` package and is not installed with it; it depends on it, by path.

## The page

**Who it is for:** the person who administers an environment's server. It is never for end users, who use their own
MCP client, and agents can never reach it: it is not an MCP server, and the decision server has no tool that changes
a status. Anyone who signs in may change the server; there is one role.

### Running it

From a checkout, with [uv](https://docs.astral.sh/uv/):

```bash
export CLM_PORTAL_KEY=<a long random key>                              # not the key agents use
set -a; . ./.postgres-local.env; set +a                                # if the environment has documents
uv run --project portal clm-portal serve examples/handbook/environment.toml
```

Then open <http://127.0.0.1:8400> and sign in with the key and your name. The name is recorded with every change you
make; it is not checked, so it is whatever you type. A wrong key waits half a second. Sessions last 12 hours and are
held in memory, so restarting the portal signs everyone out.

| Option | Default | |
|---|---|---|
| `--host` | `127.0.0.1` | Any other address prints a warning: the portal is plain HTTP. Give the address you will type in the browser, not `0.0.0.0`: a request for any other address is refused |
| `--port` | `8400` | |
| `--public-origin URL` | none | Behind a proxy that ends TLS: the address the browser uses, such as `https://portal.example.org`. A form may also come from it, and a request may name its host; at that host only this exact origin is accepted. With `https://` the session cookie is `Secure` |
| `--backend`, `--model`, `--model-url` | as the chat | Ask's starting model, changeable on the page. `LLM_API_KEY` is sent only to this `--model-url` (Ollama's by default); a URL typed on the page is called without it, and the page says so |

The portal starts **its own decision server**, `clm-server ENVIRONMENT --http` on 127.0.0.1 at a free port, with a
key made at each start and given only to that process. It never uses port 8000 or `DECISION_SERVER_KEY`, so it never
touches the server agents use. It needs the environment's folder on disk, so it runs on the server's machine.
It passes no tool selection of its own, but `clm-server` reads the file's `[serve]` table
([environment.md](../docs/environment.md#serving-settings)), so the portal's server offers the same tools as the
agents' one; Ask sees only those. If `[serve]` leaves out a collection's `document_` tool, a cited passage on the
Review page shows an error instead. If it leaves out every tool of a collection, Configuration and Documents say
the collection is not offered, rather than when it was built.

### What it shows and what it changes

- **Configuration**, read only: the environment's name and description; the command that opens its server in the MCP
  Inspector, which shows its tools, resources and prompts ([getting
  started](../docs/getting-started.md#looking-inside-the-server)); every rule,
  supplied or derived, with its origin, status, owner, version, DMN rows and links to the passages it cites; the
  information items; the graph's nodes and edges (a graph in Neo4j is named, not shown); the collections and when each
  was built; and every load error, such as why its server did not start. It is read from the files, not by loading
  the environment, so it shows them even when the server does not start.
- **Ask**: the chat in a page, with the same models and every call and result shown. As in the chat, dry run can
  hold back every tool (a tick) or only the tools you name, and **Full results** shows results whole rather than
  shortened after 1,500 characters. A name the server does not have sends nothing, so a typo cannot let a tool
  run. The conversation is kept for your session. **Testing** restarts the portal's server with `--admit-derived
  proposed,checked,approved`, so unapproved derived rules decide; while it is on, every page says so. It is off at
  every start.
- **Review**: derived decisions, facts and conflicts, each beside its cited passage (or its quote, when the server
  cannot read the passage), its check, and for a fact or conflict the supplied value and its graph row. Approve
  (only once `clm-derive check` passed it), reject, or resolve a conflict; each is recorded with your name and an
  optional note. **Correct** edits a fact's value or a decision's DMN table in place: the table must stay inside
  `derived/` and load before it is saved; the entry goes back to proposed with no check, and `clm-derive check` runs,
  so it cannot be approved until it passes again.
- **Documents**: load a collection from a `.jsonl` file in the environment's folder, or upload one, which is saved
  there (under a safe name, never over another file unless you tick overwrite) so git can hold it; this runs
  `clm-load-documents`. **Derive** runs `clm-derive run` on a collection; without `clm-derive`, the page says
  derivation is not available yet.

It changes only documents and the derived layer (`derived/derived.json`, through `clm_decision.derived`, and the
tables in `derived/`); new tools stay in files, and no supplied file is changed. Every change is written to the
files, so git stays the record. One job or change runs at a time; another asks you to wait. A job's output is shown,
with any key or password replaced by `***`, and when it succeeds the portal restarts its own server.

**The server agents use is yours to restart.** It reads the files at start, so after approving changes, restart it
(or redeploy from git) to pick them up.

**Security:** the key is in `CLM_PORTAL_KEY`, separate from the agents' key; the portal does not start with a key
shorter than 16 characters. The session cookie is `HttpOnly` and
`SameSite=Strict`; every form carries a per-session token; a POST from any other origin, or a request whose `Host`
is not the portal's address, is refused. On an address other than 127.0.0.1 the portal is plain HTTP: put it behind
TLS, and give the proxy's address in `--public-origin`.

## The chat

A small command-line chat client for MCP. Point it at a running decision server, built from an example or your own
scenario or environment, and ask questions. A model answers from the server's tools, and every tool call and its
result are printed, so you can see where an answer came from. Whether the answers are good enough is yours to judge.
Use it when you have no MCP client such as Claude, or would rather not use one. Do not give it real sensitive text
unless the model it uses may see it.

### Running it

From a checkout, with [uv](https://docs.astral.sh/uv/) and, for the starter model, [Ollama](https://ollama.com):

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

The chat sends the key in `DECISION_SERVER_KEY` (a shared key or an OAuth token) as a bearer token and never
prints it. Follow-up questions keep the conversation. `/tools` lists the tools; `/quit` leaves. Results over 1,500
characters are shortened on screen (the model gets them whole); `--full` prints them in full.

### Any model

| Backend | Option | Model's key |
|---|---|---|
| Anything that speaks the OpenAI chat API: Ollama, LM Studio, vLLM, most hosted models | `--model NAME --model-url URL` (default: `granite4:micro` on Ollama) | `LLM_API_KEY`, if it needs one |
| Claude, through the Anthropic API | `uv run --project portal clm-portal chat URL --backend claude [--model NAME]` | Anthropic's usual credentials |

The model must support tool calls. Read the printed tool results, not only the answer.

**The starter model**, `granite4:micro`, was chosen over `qwen3:4b` and `llama3.2:3b`: it was the fastest, and the
only one of the three to call `decide`. Against the same servers, three questions, once each:

| Question | `granite4:micro`, local, free | `claude-opus-5-5`, paid |
|---|---|---|
| Who supplies Product Line X's components? | Right | Right |
| Should sourcing move to Region B? | Passed the product's name, not its id `PX`; every branch failed | Right: 50%, and why 100% was rejected |
| Service desk: who handles a registry request? | Right route and ticket, but rewrote the ticket's address | Right, and did not claim more than the result showed |

What this showed:

- **A weak answer may be the model, not your server.** Every miss was the small model's; on the same tools, Claude
  answered all three. Try a larger model before changing your tools.
- **A small model shows where descriptions are unclear.** Two of the small models passed a component's name where
  its id was wanted; saying "give the id, such as C1" in the tariff descriptions fixed that for `granite4:micro`.
  Testing with a small model as well as a large one finds such gaps.
- **Speed is similar:** 4 to 7 seconds a question for `granite4:micro` on a Mac, about 10 for Claude.

### Dry runs

**By default every tool call is carried out**, including tools that act, so you can test your configuration fully.
Point acting tools at a test system if you do not want them to act for real.

- `--dry-run` prints every call and sends none.
- `--dry-run TOOL`, repeatable, holds back only those tools; the others run. A name the server does not have stops
  the chat at start-up, so a typo cannot let a tool run.

A held-back call is printed as `[dry run, not sent: TOOL {arguments}]`, and the model is told it was not carried out.

In a scenario, acting tools run inside `decide`, so holding back `decide` skips the whole decision: you see that it
was called, not the route. To see decisions without real actions, point the acting tools at a stand-in, as the
service desk's `mock_tracker.py` does.

## Tests

`uv run --project portal pytest portal/tests` runs against a stand-in server and model; it needs no network,
key or PostgreSQL.
