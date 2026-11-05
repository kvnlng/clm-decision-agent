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
| [portal](../portal/README.md) | Administering a server from a local page, and chatting with a running server |

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
| Examples | [service desk](../examples/service_desk/README.md), [CrowdStrike day](../examples/crowdstrike/README.md) | [tariff](../examples/tariff/README.md), [new site](../examples/new_site/README.md), [non-competes](../examples/noncompete/README.md), [handbook](../examples/handbook/README.md) |
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

**Without installing**, `uvx` runs a command straight from GitHub, in a cached environment of its own; an MCP client
or the Inspector can start a server this way:

```bash
uvx --from "git+https://github.com/kvnlng/clm-decision-agent" clm-server my_environment/environment.toml
uvx --from "clm-decision-agent[documents] @ git+https://github.com/kvnlng/clm-decision-agent" clm-server ...  # an extra
```

| Extra | Adds |
|---|---|
| `clm` | CLM: scenarios, judges, training and the example writer |
| `neo4j` | The Neo4j driver, for a graph in Neo4j |
| `documents` | The PostgreSQL driver and pgvector, for collections of documents |
| `agent` | The Anthropic SDK, for `clm-agent` with Claude; Ollama needs nothing more |

Commands: `clm-server`, `clm-agent`, `clm-new-scenario`, `clm-new-environment`,
`clm-synthesize`, `clm-train`, `clm-evaluate`, `clm-record`, `clm-load-graph`, `clm-load-documents`, `clm-registry`,
`clm-trace-report`, `clm-derive`. Each takes `--help`, and one that needs a missing extra says which. In Python, the package is
`clm_decision`.

## Try an example

[examples/README.md](../examples/README.md) lists all six.

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
npx -y @modelcontextprotocol/inspector@2.13.0 \
    uv run clm-server examples/service_desk/scenario.toml --review    # terminal 2: the MCP Inspector
```

In the Inspector ([Looking inside the server](#looking-inside-the-server)), call `decide` with a message such as "I
need access to the oncology registry for an audit, needed this week". The result has the route (Data_Access) with its
probabilities, each question's answer, the facts the informing tools found, the rules that apply, and the ticket the
acting tool opened in the mock tracker: tools run for real. In review mode, `correct` saves the right route for a
message and `agreement` says how often CLM agrees with your corrections.

To try `decide` without opening tickets, start the server with `--read-only` instead of `--review` (the two do not go
together, since `correct` writes the record): the informing tools still run, and each acting tool is reported
`skipped` with the reason "read-only server". The mock tracker is then not needed.

## Build your own

- **Routing:** `uv run clm-new-scenario my_scenario`, then the [walkthrough](walkthrough.md). In review mode, the `correct`
  tool saves a correction; [fine-tuning](fine-tuning.md) trains CLM on them.
- **Weighing options:** `uv run clm-new-environment my_environment`, then [environment](environment.md). Its graph can
  live in [Neo4j](environment.md#on-neo4j).
- **Either:** the MCP Inspector shows everything the server offers and calls one tool at a time, with no agent; see
  [Looking inside the server](#looking-inside-the-server).

A mistake in either file stops start-up with one line naming the setting.

## Looking inside the server

The [MCP Inspector](https://modelcontextprotocol.io/docs/tools/inspector), from the MCP project, is a page in your
browser that starts your server, connects to it and shows everything it offers: its tools, its resources and its
prompts. You can call one tool at a time and read the result, with no model and no agent. Use it after every change
to your environment, before trying an agent. It needs [Node](https://nodejs.org); `npx` downloads the Inspector the
first time.

**1. Start it** from the repository, naming the environment. The tariff example needs nothing else:

```bash
npx -y @modelcontextprotocol/inspector@2.13.0 uv run clm-server examples/tariff/environment.toml
```

It prints an address such as `http://127.0.0.1:6274/?MCP_INSPECTOR_API_TOKEN=...` and opens it in your browser. The
token in the address lets only you use that page; open the address as printed.

**2. Connect.** The page shows a server card with the command it will run. Turn on its switch: the dot beside it goes
from **Disconnected** to **Connected**, and the server's name, `decision-server`, appears at the top with four tabs:
**Servers**, **Tools**, **Prompts** and **Resources**. If it shows **Failed**, the **Console** on the right has the
server's one-line reason, such as a mistake in the environment file.

**3. Tools.** Open **Tools**. The list has every tool the server offers (18 for tariff). Choose one, for
example `query_components_of`: its description, whether it is read-only, and a box for each input appear.
Type `PX` as the product and press **Execute Tool**. The result is the JSON an agent would get: the rows, and the
steps that produced them.

**4. Resources.** Open **Resources**. Under **URIs** are the things a client can read without calling a tool: for
tariff, `describe`; for an environment with rules and documents, each rule and each document as well. Under
**Templates** are the forms for reading any one of them, such as one passage of a document. Choose one to see what it
holds.

**5. Prompts.** Open **Prompts**. `answer_question` is the reference agent's instructions with your environment's
description; give it a question to see the message an agent would start from.

**6. Change and look again.** After you edit the environment, turn the switch off and on: the Inspector starts the
server again from your files. The **Protocol** panel on the right lists every message between the page and the
server, with how long each took.

**An environment with documents** also needs the password to its PostgreSQL, and the Inspector passes the server none
of your shell's variables. Start the server through a shell that reads them from your file, so the password never
appears in the command or on the page:

```bash
npx -y @modelcontextprotocol/inspector@2.13.0 sh -c \
    'set -a; . ./.postgres-local.env; exec uv run clm-server examples/handbook/environment.toml'
```

The handbook shows 22 tools, 19 resources (`describe`, 5 rules and 13 documents), 3 templates and 1 prompt.

**A routing scenario** works the same way: give the scenario file, and `--review` for its `correct` and `agreement`
tools (`npx -y @modelcontextprotocol/inspector@2.13.0 uv run clm-server my_scenario.toml --review`). Call `decide`
with a message's `text`; its acting tools run for real.

**One call from the command line**, for a check you can repeat or put in a script, is `--cli` with a method:

```bash
npx -y @modelcontextprotocol/inspector@2.13.0 --cli uv run clm-server examples/tariff/environment.toml \
    --method tools/call --tool-name scenario_sourcing --tool-arg product=PX
```

It prints the result as JSON. `--method tools/list`, `resources/list` and `prompts/list` list them. A server already
running over HTTP is reached with its address and key instead: [serving](serving.md#inspecting-a-server). The
Inspector's page changes between versions; these steps are for 2.13.0.

## Trying it in a chat

With no MCP client of your own, or none you want to use, the [portal](../portal/README.md)'s chat lets you talk to a
running server from the command line. A model answers from your tools and every call and result is printed, so you
can see where each answer came from and judge it. The portal is not part of the package; its page also shows an
environment's whole configuration and reviews derived rules.

```bash
ollama pull granite4:micro                                            # the small starter model, about 2.1 GB
export DECISION_SERVER_KEY=<a long random key>
uv run clm-server examples/tariff/environment.toml --http             # terminal 1
uv run --project portal clm-portal chat http://127.0.0.1:8000/mcp     # terminal 2
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
- **`clm-agent`** takes a server's URL as well as a file, with the key in `DECISION_SERVER_KEY`,
  sent only over https or to this computer.
- **Databricks:** not tried yet; [what it needs](serving.md#databricks).

**A scenario's server acts without asking.** An acting tool runs at once unless the route is uncertain, and calling
`decide` twice acts twice, so point acting tools at real systems with care. An environment's tools only read. Read
[data handling](data-handling.md) before any real text goes in.
