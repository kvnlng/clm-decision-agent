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

## The server's tools

For a scenario:

| MCP tool | What it does |
|---|---|
| `decide(text, context)` | The route with its probabilities, `uncertain` below the threshold, each question's answer, the rules that apply, the facts found, the option chosen, and what the acting tools did |
| `decide_many(requests)` | The same for up to 1000 requests at once, faster per request |
| `check_call(text, call)` | CLM's opinion of a tool call the caller proposes: a probability and approved yes or no |
| `describe()` | The routes, options, questions, rules and tools |
| `correct(...)`, `agreement()` | Review mode only: save the right decision; how often CLM agrees with the corrections |

For an environment:

| MCP tool | What it does |
|---|---|
| `describe` | The environment and every part, with its parameters, levers, objective and limits |
| `query_<name>`, `decision_<name>`, `calculation_<name>` | One named part, with typed arguments |
| `scenario_<name>` | Every branch, or those asked for: figures, kept or rejected, the limits that rejected it, and every step |
| `judge_<name>` | CLM's probability that the judge's question is answered yes for a text |
| `rules`, `rule` | When the rules carry metadata: those governing a topic, with why each exists and what to ask for; one rule's rows |

## What to expect

Measured on an M5 Max with 64 GB unless a GPU is named. Small tests, said plainly.

- **CLM must be trained.** On the service desk the released heads routed 51% of held-back examples right; trained
  on 199 generated examples, 98%.
- **Against Claude, on the CrowdStrike day's 600 check cases, Claude was more accurate and, on the Mac, faster.**
  Sonnet 5.5 got 82% of routes right and Haiku 5.5 79%, against CLM's 75%; each took about 267 ms a decision to
  CLM's 300 ms. CLM's strengths are cost and place: $0 for the day's 99,806 decisions against about $15 (Haiku) or
  $158 (Sonnet), the messages never leave the machine, and it marked 119 of its 152 wrong routes uncertain, so they
  went to a person ([the comparison](examples/crowdstrike/README.md)).
- **On a data-centre GPU CLM is much faster:** 3.6 decisions a second on the Mac, about 19 on an A100 and 50 on an
  H100, against a peak need of 10.7 on the CrowdStrike day ([docs/findings.md](docs/findings.md)).
- **CLM's probabilities are relative, not calibrated.** Tune thresholds on labelled cases; the tariff news judge
  separated risks from harmless news by a gap of only 0.05.
- **In the tariff example, Claude and a local `gemma4:26b` reached the same decision** by the same two calls in
  eight runs. Claude's answers were fuller and explained the dependency; once it worked out a figure itself, which
  the number check flagged. gemma4 was free and stayed on the Mac, but with news available it did not run the
  scenario that checks it and missed the strike ([the runs](examples/tariff/README.md#what-happened-when-the-models-were-asked)).

## Layout

The framework is the package `clm_decision`, in `src/clm_decision/`:

| Module | Purpose |
|---|---|
| `server.py` | `clm-server`: MCP over stdio or HTTP, for a scenario or an environment |
| `scenario.py`, `decision.py`, `tools.py` | Reads a scenario; one decision; loads its tools |
| `environment.py`, `environment_server.py`, `engine.py` | Reads an environment; its parts as MCP tools; queries, decisions, scenarios and judges run with every step recorded |
| `graph.py`, `graph_neo4j.py`, `load_graph.py`, `dmn.py` | The built-in graph; named queries on Neo4j; `clm-load-graph`; DMN decision tables |
| `decision_agent.py`, `number_check.py`, `decision_record.py` | `clm-agent`, Claude or a local model; the check that every figure came from a tool; the decision record |
| `call.py` | `clm-call`: a server's tools, listed or called by hand, over stdio or HTTP |
| `oauth.py`, `registry.py` | OAuth tokens from any OIDC issuer; `clm-registry`, the `server.json` for an MCP registry |
| `tracing.py`, `trace_report.py` | `--trace`, and `clm-trace-report` |
| `chat.py` | `clm-chat`, the test chat for scenarios |
| `clm_router.py`, `clm_torch.py`, `clm_schema.py`, `clm_common.py` | CLM with MLX or PyTorch; routing and the reuse of corrections; the authors' question format, unchanged |
| `record.py`, `evaluate.py`, `new_scenario.py`, `new_environment.py` | Corrections (`clm-record`) and their scoring (`clm-evaluate`); `clm-new-scenario`, `clm-new-environment` |
| `dataset.py`, `synthesize.py`, `train.py`, `generator.py` | `clm-train`; `clm-synthesize` and the language model that writes examples |

Outside the package:

| Folder | Purpose |
|---|---|
| `examples/` | The worked examples: `service_desk/`, `crowdstrike/`, `tariff/` and `new_site/` |
| `docs/` | Guides: getting started, scenario, walkthrough, environment, serving, registry, fine-tuning, models, findings, data handling |
| `tests/` | `uv run pytest` (no models needed); `uv run pytest -m slow` (loads CLM; `CLM_TEST_BACKEND=torch` for torch) |

## Licence

Copyright 2026 Kevin Long. Licensed under the [Apache License 2.0](LICENSE). The licence covers this repository only:
the models it downloads have their own terms.
