# Getting started

A prototype, not used in production and **not for clinical decisions**: the service desk example routes questions
about patient care out of scope.

## All the guides

| Guide | What it covers |
|---|---|
| [environment](environment.md) | An environment for weighing options; its keys, limits and figures in the [reference](environment-reference.md) |
| [scenario](scenario.md) | A scenario for routing; its keys and exact rules in the [reference](scenario-reference.md) |
| [walkthrough](walkthrough.md) | A scenario, step by step |
| [fine-tuning](fine-tuning.md) | Training CLM on your routes |
| [models](models.md) | CLM and language models, and where they run |
| [serving](serving.md) | Serving decisions to agents; the flags, OAuth setups and measurements in the [reference](serving-reference.md) |
| [registry](registry.md) | Listing a deployment in an MCP registry |
| [data handling](data-handling.md) | Where text goes; read before using real text |
| [findings](findings.md) | What the benchmark traces showed |
| [portal](portal.md) | An admin page `clm-server` serves, and a chat with a running server |

## Your environment, and the decisions over it

**An environment is what you define:** queries over your data, actions such as opening a ticket, and rules: DMN
tables, calculations and limits. The framework runs them, records each result and traces every step. Decisions over
an environment come in two styles:

| | **Routing** | **Weighing options** |
|---|---|---|
| Decides | A message's route, many at once | A question with alternatives: "move Product Line X to Region B?" |
| How | CLM picks the route from the message and your informing tools' facts; your acting tools carry it out | Queries supply figures, calculations work out each alternative, rules reject those over a limit and an objective ranks the rest |
| CLM | Ranks the options | Only in a **judge**: a yes/no question of text no rule can read |
| The agent | Talks to the person, acts on the decision | Chooses what to run and reports it; never calculates; every figure is checked |
| File | `scenario.toml` | `environment.toml` |
| Examples | [service desk](../examples/service_desk/README.md), [CrowdStrike day](../examples/crowdstrike/README.md) | [tariff](../examples/tariff/README.md), [new site](../examples/new_site/README.md), [non-competes](../examples/noncompete/README.md), [handbook](../examples/handbook/README.md) |
| Guide | [scenario](scenario.md), [walkthrough](walkthrough.md) | [environment](environment.md) |

`clm-server` serves one file at a time, so the two styles cannot yet share tools. A CLM does not generate text: it
says which listed option fits a situation, with a probability (Kwok et al. 2026,
[Contrastive-LM/CLM](https://github.com/Contrastive-LM/CLM)). This project runs the authors' model with PyTorch, or
community MLX builds on a Mac.

## Installing

You need Python 3.12 to 3.14, [uv](https://docs.astral.sh/uv/), and for CLM one of:

| Hardware | Backend | First run downloads | |
|---|---|---|---|
| Mac, Apple Silicon, 16 GB or more | MLX | about 8 GB | |
| Linux, NVIDIA GPU of 24 GB or more | PyTorch | about 16 GB | Set the torch backend in the scenario's `[clm]` table; install a C compiler and Python's headers (`sudo apt install build-essential python3.12-dev` on Ubuntu). A CPU works, slowly: [serving](serving.md#on-linux) |

In a checkout, run `uv sync`, then `uv run <command>`. Installed as a package, the core (about 50 MB) serves
environments without judges; `uvx` runs it uninstalled, so an MCP client or the Inspector can start a server:

```bash
pip install "clm-decision-agent @ git+https://github.com/kvnlng/clm-decision-agent"               # the core
pip install "clm-decision-agent[clm,neo4j,agent] @ git+https://github.com/kvnlng/clm-decision-agent"
uv tool install "clm-decision-agent[clm] @ git+https://github.com/kvnlng/clm-decision-agent"       # commands only
uvx --from "git+https://github.com/kvnlng/clm-decision-agent" clm-server my_environment/environment.toml
uvx --from "clm-decision-agent[documents] @ git+https://github.com/kvnlng/clm-decision-agent" clm-server ...  # an extra
```

| Extra | Adds |
|---|---|
| `clm` | CLM: scenarios, judges, training and the example writer |
| `neo4j` | The Neo4j driver, for a graph in Neo4j |
| `documents` | The PostgreSQL driver and pgvector, for collections of documents |
| `agent` | The Anthropic SDK, for `clm-agent` with Claude; Ollama needs nothing more |

The commands are `clm-server`, `clm-agent`, `clm-new-scenario`, `clm-new-environment`, `clm-synthesize`,
`clm-train`, `clm-evaluate`, `clm-record`, `clm-load-graph`, `clm-load-documents`, `clm-registry`,
`clm-trace-report` and `clm-derive`. Each has `--help`, and one that needs a missing extra says which. The Python
package is `clm_decision`.

## Try an example

[examples/README.md](../examples/README.md) lists all six. **Tariff**, weighing options, needs no training:

```bash
uv run clm-server examples/tariff/environment.toml               # serve it over MCP, for any agent
uv run clm-agent examples/tariff/environment.toml \
    "Should we shift Product Line X's sourcing to Region B, and what would it do to Q3 margin and SLA times?"
```

`clm-agent` is a reference agent, Claude by default (your usual Anthropic credentials) or `--backend ollama`
(`gemma4:26b`). It answers from the server's tools, then lists any figure no tool returned.

**Service desk**, routing, is trained first, in under a minute, since as released CLM gets about half the routes
right:

```bash
uv run clm-train examples/service_desk/scenario.toml
uv run python examples/service_desk/mock_tracker.py              # terminal 1: a stand-in issue tracker
npx -y @modelcontextprotocol/inspector@2.13.0 \
    uv run clm-server examples/service_desk/scenario.toml --review    # terminal 2: the MCP Inspector
```

In the Inspector ([Looking inside the server](#looking-inside-the-server)), call `decide` with `text` "I need access
to the oncology registry for an audit, needed this week". The result shows the route (Data_Access) and its
probabilities, each question's answer, the informing tools' facts, the rules, and the ticket really opened in the
mock tracker. `--review` adds `correct`, which saves the right route, and `agreement`, which reports how often CLM
agrees with your corrections. With `--read-only` instead (not both, since `correct` writes the record), acting tools
are reported `skipped` with the reason "read-only server", so the tracker is not needed.

## Build your own

- **Routing:** `uv run clm-new-scenario my_scenario`, then the [walkthrough](walkthrough.md);
  [fine-tuning](fine-tuning.md) trains CLM on corrections.
- **Weighing options:** `uv run clm-new-environment my_environment`, then [environment](environment.md); the graph
  can live in [Neo4j](environment.md#on-neo4j).

A mistake in either file stops start-up with one line naming the setting.

## Looking inside the server

The [MCP Inspector](https://modelcontextprotocol.io/docs/tools/inspector) is a browser page that starts your server
and shows its tools, resources and prompts, calling one tool at a time with no model and no agent; use it after
every change, before an agent. It needs [Node](https://nodejs.org); `npx` downloads it. These steps are for 2.13.0;
the page changes between versions.

```bash
npx -y @modelcontextprotocol/inspector@2.13.0 uv run clm-server examples/tariff/environment.toml
```

1. **Start it** from the repository: it opens `http://127.0.0.1:6274/?MCP_INSPECTOR_API_TOKEN=...`, whose token
   lets only you use the page.
2. **Connect** with the server card's switch; on **Failed**, the **Console** has the one-line reason.
3. **Tools** lists every tool (18 for tariff): run `query_components_of` with `PX` to see the JSON an agent gets,
   the rows and the steps behind them.
4. **Resources** are read without a tool: `describe`; **Templates** read one rule, document or passage by name.
5. **Prompts** has `answer_question`, the reference agent's instructions with your environment's description.
6. **After an edit**, switch off and on to restart from your files; the **Protocol** panel times every message.

**An environment with documents** needs its PostgreSQL password, and the Inspector passes on none of your shell's
variables, so start the server through a shell that reads your file; the password never reaches the command or the
page. The handbook shows 22 tools, 1 resource (`describe`), 3 templates and 1 prompt:

```bash
npx -y @modelcontextprotocol/inspector@2.13.0 sh -c \
    'set -a; . ./.postgres-local.env; exec uv run clm-server examples/handbook/environment.toml'
```

**One call from the command line** is `--cli` with a method, and the result is printed as JSON; `--method
tools/list`, `resources/list` and `prompts/list` list the tools, resources and prompts. A server already on HTTP is
reached by address and key:
[serving](serving.md#inspecting-a-server).

```bash
npx -y @modelcontextprotocol/inspector@2.13.0 --cli uv run clm-server examples/tariff/environment.toml \
    --method tools/call --tool-name scenario_sourcing --tool-arg product=PX
```

## Trying it in a chat

With no MCP client of your own, the [portal](portal.md)'s chat, `clm-chat`, talks to a running server from the
command line through any OpenAI-style model or Claude, printing every call and result. Its options are in the
portal's page.

## Connecting agents

```bash
export DECISION_SERVER_KEY=<a long random key>
uv run clm-server my_scenario/scenario.toml --http                   # a shared key
uv run clm-server examples/tariff/environment.toml --http --auth oauth \
    --issuer https://login.example.org/realms/corp --audience decisions   # tokens from your identity provider
```

The rest, in detail:

| | |
|---|---|
| HTTP, shared key or OAuth from any OIDC issuer | [serving](serving.md) |
| An MCP registry, so agents find a deployment | `clm-registry`, [registry](registry.md) |
| Tracing | `--trace FILE` records each call's time and caller; `clm-trace-report` summarises it |
| `clm-agent` | Takes a server's URL as well as a file; the key in `DECISION_SERVER_KEY` goes only over https or to this computer |
| Databricks | Not tried yet; [what it needs](serving.md#databricks) |

**A scenario's server acts without asking:** an acting tool runs at once unless the route is uncertain, and calling
`decide` twice acts twice, so point acting tools at real systems with care. An environment's tools only read. Read
[data handling](data-handling.md) before any real text goes in.
