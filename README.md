# CLM decision server

**This is a framework, not a finished product.** It supports building the tools an AI agent uses to reach the
resources where an organisation's information already lives (databases, knowledge graphs, rule tables, issue
trackers, public data) and to decide from them. You write those tools and the rules that combine them; the framework
serves them over [MCP](https://modelcontextprotocol.io) as a **decision server**, runs them, checks them and traces
every step. It holds no rules or data of its own, and decides nothing until you write the decision down.

Use it when a decision must rest on your information and your rules, not an assistant's guess: one made thousands
of times a day, such as which team handles a request, or one made once, such as whether to change supplier or open a
new site. The agent, whether Claude, a local model or one built on Databricks, talks to the person; the tools supply the
information, and the decision comes back with its reasons.

<p align="center"><img src="docs/images/architecture.svg" alt="Any MCP agent (Claude, a local model, Databricks) calls clm-server over MCP. The server runs decisions, by routing with CLM or by weighing options against rules, over your environment: the tools you define, which reach your data, take actions and hold your rules." width="820"></p>

The framework holds you to three principles:

- **You write the rules; agents only choose.** Information reaches a decision only through your tools.
- **Every decision is checkable.** Every step is traced, every figure checked, and a decision record kept.
- **Small and local where you can.** CLM picks among options on a Mac or one GPU: cheap, and your data stays
  with you. Claude was more accurate in the benchmark.

**[Getting started](docs/getting-started.md)** covers your environment and the two styles of decision, installing,
the examples, building your own and connecting agents. It is a prototype, not used in production and **not for
clinical decisions**; a scenario's server runs its acting tools without asking.

## Tools

These are the MCP tools a decision server offers an agent. Which ones depends on the style of decision.

### Routing: a scenario (`scenario.toml`)

A fixed set. The agent asks for a decision; your informing and acting tools run inside `decide`, not as tools of their own.

| MCP tool | What it does |
|---|---|
| `decide(text, context)` | The route with its probabilities, `uncertain` below the threshold, each question's answer, the rules that apply, the facts found, the option chosen, and what the acting tools did |
| `decide_many(requests)` | The same for up to 1000 requests at once, faster per request |
| `check_call(text, call)` | CLM's opinion of a tool call the caller proposes: a probability and approved yes or no |
| `describe()` | The routes, options, questions, rules and tools |
| `correct(...)`, `agreement()` | Review mode only: save the right decision; how often CLM agrees with the corrections |

### Weighing options: an environment (`environment.toml`)

One tool for each part you write, named after it, so the agent can explore the parts and run the ones it needs.

| MCP tool | What it does |
|---|---|
| `describe` | The environment and every part, with its parameters, levers, objective and limits |
| `query_<name>`, `decision_<name>`, `calculation_<name>` | One named part, with typed arguments |
| `scenario_<name>` | Every branch, or those asked for: figures, kept or rejected, the limits that rejected it, and every step |
| `judge_<name>` | CLM's probability that the judge's question is answered yes for a text |
| `search_<name>`, `document_<name>`, `cites_<name>`, `cited_by_<name>` | A collection of documents: the passages closest to a text, one document as stored, and the citations between them |
| `rules`, `rule` | When the rules carry metadata: those governing a topic, with why each exists and what to ask for; one rule's rows |

## What to expect

Small tests, said plainly ([docs/findings.md](docs/findings.md)):

- **Train CLM first:** 51% of routes right as released, 98% after training on the service desk.
- **Claude is more accurate:** 82% (Sonnet) against CLM's 75% on the CrowdStrike day, and faster on a Mac. CLM costs
  nothing to run, keeps data local and sends most of its mistakes to a person.
- **A GPU makes CLM fast:** about 19 decisions a second on an A100 and 50 on an H100, against 3.6 on a Mac.
- **CLM's probabilities are relative:** tune thresholds on labelled cases.

## Layout

| Folder | Holds |
|---|---|
| `src/clm_decision/` | The framework, and the `clm-*` commands |
| `examples/` | Four worked examples ([the list](examples/README.md)) |
| `docs/` | The guides, starting with [getting started](docs/getting-started.md) |
| `client/` | A command-line [test client](client/README.md), not part of the package |
| `tests/` | `uv run pytest`; `uv run pytest -m slow` also loads CLM (`CLM_TEST_BACKEND=torch` for torch) |

## Licence

Copyright 2026 Kevin Long. Licensed under the [Apache License 2.0](LICENSE). The licence covers this repository only:
the models it downloads have their own terms.
