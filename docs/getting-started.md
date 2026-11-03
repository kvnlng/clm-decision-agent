# Getting started

A prototype, not used in production and **not for clinical decisions**: the service desk example routes questions
about patient care out of scope.

## All the guides

| Guide | What it covers |
|---|---|
| [environment](environment.md) | Writing an environment for weighing options |
| [scenario](scenario.md) | Writing a scenario for routing |
| [walkthrough](walkthrough.md) | Building a scenario step by step |
| [fine-tuning](fine-tuning.md) | Training CLM on your routes |
| [models](models.md) | Choosing CLM and language models, and where they run |
| [serving](serving.md) | Serving decisions to agents |
| [registry](registry.md) | Listing a deployment in an MCP registry |
| [data handling](data-handling.md) | Where text goes; read before using real text |
| [findings](findings.md) | What the benchmark traces showed |
| [test client](../client/README.md) | Chatting with a running server from the command line, for testing |

## Your environment, and the decisions over it

**Your environment is the tools you define:** queries that reach your data, actions such as opening a ticket, and your
rules (DMN tables, calculations, limits). The framework runs them, records what each returned and traces every step.

**A decision runs over your environment in one of two styles:**

| | **Routing** | **Weighing options** |
|---|---|---|
| Decides | Which route a message takes, many at once | A question with alternatives: "should we move Product Line X to Region B?" |
| How | CLM picks the route from the message and your informing tools' facts; your acting tools carry it out | Your queries supply figures, calculations work out each alternative, rules reject those over a limit, an objective ranks the rest |
| CLM | Ranks the options | Only in a **judge**: a yes/no question of text no rule can read |
| The agent | Talks to the person and acts on the decision | Chooses what to run and reports it; never calculates, and every figure is checked |
| File | `scenario.toml` | `environment.toml` |
| Examples | [service desk](../examples/service_desk/README.md), [CrowdStrike day](../examples/crowdstrike/README.md) | [tariff](../examples/tariff/README.md), [new site](../examples/new_site/README.md) |
| Guide | [scenario](scenario.md), [walkthrough](walkthrough.md) | [environment](environment.md) |

Each style has its own file today, and `clm-server` serves one file at a time, so the two cannot yet share tools.

A CLM does not generate text: it says which of a list of options fits a situation, with a probability. It is described
in Kwok et al. (2026) and published at [Contrastive-LM/CLM](https://github.com/Contrastive-LM/CLM); this project runs
the authors' model with PyTorch, or community MLX builds on a Mac.

## Installing

You need Python 3.12 to 3.14, [uv](https://docs.astral.sh/uv/), and for CLM one of:

- **A Mac with Apple Silicon** and 16 GB or more. Uses MLX; the first run downloads about 8 GB.
- **Linux with an NVIDIA GPU of 24 GB or more.** Uses PyTorch; the first run downloads about 16 GB. Set the torch
  backend in the scenario's `[clm]` table, and install a C compiler and Python's headers
  (`sudo apt install build-essential python3.12-dev` on Ubuntu). A CPU works, slowly. See [serving](serving.md#on-linux).

**In a checkout:** `uv sync`, then `uv run <command>`.

**As a package**, for example in a Databricks notebook. The core is about 50 MB and serves environments without judges:

```bash
pip install "clm-decision-agent @ git+https://github.com/kvnlng/clm-decision-agent"               # the core
pip install "clm-decision-agent[clm,neo4j,agent] @ git+https://github.com/kvnlng/clm-decision-agent"
uv tool install "clm-decision-agent[clm] @ git+https://github.com/kvnlng/clm-decision-agent"       # commands only
```

| Extra | Adds |
|---|---|
| `clm` | CLM: scenarios, judges, training and the example writer |
| `neo4j` | The Neo4j driver, for a graph in Neo4j |
| `documents` | The PostgreSQL driver and pgvector, for collections of documents |
| `agent` | The Anthropic SDK, for `clm-agent` with Claude; Ollama needs nothing more |

Commands: `clm-server`, `clm-agent`, `clm-chat`, `clm-call`, `clm-new-scenario`, `clm-new-environment`,
`clm-synthesize`, `clm-train`, `clm-evaluate`, `clm-record`, `clm-load-graph`, `clm-load-documents`, `clm-registry`,
`clm-trace-report`. Each takes `--help`, and one that needs a missing extra says which. In Python, the package is
`clm_decision`.

## Try an example

[examples/README.md](../examples/README.md) lists all four.

**Weighing options: tariff.** No training needed.

```bash
uv run clm-server examples/tariff/environment.toml               # serve it over MCP, for any agent
uv run clm-agent examples/tariff/environment.toml \
    "Should we shift Product Line X's sourcing to Region B, and what would it do to Q3 margin and SLA times?"
```

`clm-agent` is a reference agent: Claude by default (your usual Anthropic credentials), or a local model with
`--backend ollama` (`gemma4:26b`). It answers from the server's tools, then lists any figure no tool returned.

**Routing: service desk.** Train first, under a minute; as released, CLM gets about half the routes right.

```bash
uv run clm-train examples/service_desk/scenario.toml
uv run python examples/service_desk/mock_tracker.py              # terminal 1: a stand-in issue tracker
uv run clm-chat examples/service_desk/scenario.toml --review     # terminal 2: the test chat
```

```
you> I need access to the oncology registry for an audit, needed this week
  route: Data_Access 0.65 (next: Information_Request 0.23)
  answers: urgency 1.61   identifiable 0.54
  facts: lookup_queue: Data Governance handles this kind of request. ...
  rules: urgency >= 1.5: Set the ticket's priority to High and state the deadline in the summary.
  actions: create_ticket done: {"number": 5, "url": "http://127.0.0.1:8765/..."}
```

## Build your own

- **Routing:** `uv run clm-new-scenario my_scenario`, then the [walkthrough](walkthrough.md). In review mode, `/correct
  <route>` saves a correction; [fine-tuning](fine-tuning.md) trains CLM on them.
- **Weighing options:** `uv run clm-new-environment my_environment`, then [environment](environment.md). Its graph can
  live in [Neo4j](environment.md#on-neo4j).
- **Either:** `uv run clm-call <file>` lists the tools, and `uv run clm-call <file> <tool> '<json>'` calls one, with
  no agent. Give a running server's URL instead of a file to reach it.

A mistake in either file stops start-up with one line naming the setting.

## Testing it with the test client

With no MCP client of your own, or none you want to use, the [test client](../client/README.md) lets you chat with a
running server from the command line. A model answers from your tools and every call and result is printed, so you
can see where each answer came from and judge it. It is for testing only and is not part of the package.

```bash
ollama pull granite4:micro                                            # the small starter model, about 2.1 GB
export DECISION_SERVER_KEY=<a long random key>
uv run clm-server examples/tariff/environment.toml --http             # terminal 1
uv run --project client clm-test-client http://127.0.0.1:8000/mcp     # terminal 2
```

- **Any model:** anything that speaks the OpenAI chat API, or Claude with `--backend claude`. A weak answer may be
  the small model, not your server; try a larger one before changing your tools.
- **Tools run for real** by default, acting ones included. `--dry-run` shows calls without sending them;
  `--dry-run TOOL` holds back only that tool.

## Connecting agents

```bash
export DECISION_SERVER_KEY=<a long random key>
uv run clm-server my_scenario/scenario.toml --http                   # a shared key
uv run clm-server examples/tariff/environment.toml --http --auth oauth \
    --issuer https://login.example.org/realms/corp --audience decisions   # tokens from your identity provider
```

- **HTTP** with a shared key or OAuth from any OIDC issuer: [serving](serving.md).
- **An MCP registry**, so agents can find a deployment: `clm-registry` ([registry](registry.md)).
- **Tracing:** `--trace FILE` records each call's time and caller; `clm-trace-report` summarises it.
- **`clm-agent`** takes a server's URL as well as a file, with the key in `DECISION_SERVER_KEY`.
- **Databricks:** not tried yet; [what it needs](serving.md#databricks).

**A scenario's server acts without asking.** An acting tool runs at once unless the route is uncertain, and calling
`decide` twice acts twice, so point acting tools at real systems with care. An environment's tools only read. Read
[data handling](data-handling.md) before any real text goes in.
