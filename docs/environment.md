# Writing an environment

An environment is what you define: queries over your data, rules, calculations and the alternatives to weigh. Decisions
over an environment come in two styles: routing a message with CLM ([scenario.md](scenario.md)), or weighing options
against your rules, set out in `environment.toml`, which this guide describes. Weighing suits decisions that rest on an
organisation's data, with arithmetic, policy and alternatives to compare: "should we move Product Line X's sourcing to
Region B, and what would it do to the Q3 margin and our delivery terms?".

Start with `uv run clm-new-environment my_environment`: a working offsite-planning environment to edit into your own.
[examples/tariff](../examples/tariff/README.md) uses everything here.

## Four parts, and one rule

| Part | Answers | The owner writes |
|---|---|---|
| Knowledge graph | What exists and how it connects | The data, and named queries over it |
| Decision models | What the rules, formulas and limits say | DMN tables, or rule functions |
| Scenarios | Which alternatives there are, and which is best | Levers and allowed values, objective, limits |
| Agent | Turns a question into calls, and results into an answer | Nothing: any MCP client; `clm-agent` is a reference one |

**The owner writes every query, rule, calculation and lever; the agent only chooses among them.** It never writes a
graph query, states a rule or invents a lever value, and is told never to calculate, so every figure and verdict
traces to a step, and `number_check.py` checks the answer against those steps. A rule the owner does not supply can
be derived from the documents and checked before a person approves it, and each rule's kind is never hidden
([Derived rules and facts](#derived-rules-and-facts)). Without judges the server has no language model: the same
inputs give the same outputs. A judge adds one, for evidence no rule can read, and every answer it gives is traced.

## The file

Put `environment.toml` in a folder with the graph's CSV files, the DMN tables and your Python module. A mistake stops
loading with one line naming the setting.

```toml
[environment]
name = "Component sourcing"
description = "What the agent reads first: what this is for and where to start."
functions = "tariff_functions"            # tariff_functions.py beside this file

[graph]
backend = "builtin"                       # or "neo4j": see "On Neo4j" below
nodes = "graph/nodes.csv"
edges = "graph/edges.csv"
```

Every table of the file and what it declares: [reference](environment-reference.md#the-file).

`clm-server` serves a file with an `[environment]` table without loading CLM. The MCP Inspector starts it and shows
its tools, resources and prompts, and calls one, with no agent
([getting started](getting-started.md#looking-inside-the-server); the commands are in
[the walkthrough](#from-a-new-folder-to-a-registered-server)).

```bash
uv run clm-server examples/tariff/environment.toml            # MCP over stdio
DECISION_SERVER_KEY=... uv run clm-server examples/tariff/environment.toml --http
```

`--http`, the key and `--trace` work as in routing ([serving.md](serving.md)). Review mode is for routing only.

## The graph

`nodes.csv` has the columns `id,kind,name` and any others; `edges.csv` has `source,relation,target` and any others.
Extra columns are properties: an empty cell is none, a plain decimal (`12`, `-3`, `0.015`) is a number, and `1e5`,
`1_000`, `nan` and `inf` stay text. An edge to an unknown node stops loading, naming the file and line. The
operations named queries use: [reference](environment-reference.md#the-graph).

## Named queries

```toml
[queries.suppliers_of]
description = "Who supplies a component: region, unit price, lead time in days, defect rate and freight per unit."
params = { component = "string" }         # string, number, integer, boolean or list
function = "suppliers_of"                 # suppliers_of(graph, component) -> a list of dicts
```

**A query must fail when what it was asked about does not exist.** No rows looks like an answer: an unknown team had
nobody in it, so every venue fitted and the cheapest won. Raise an error (`LookupError(f"no team {team}")`) and the
branch fails, naming it. What to put in the description and the rows:
[reference](environment-reference.md#named-queries).

## On Neo4j

Named queries can run on Neo4j, with the Cypher beside each query's function
([reference](environment-reference.md#on-neo4j)). The driver is the `neo4j` extra
(`pip install 'clm-decision-agent[neo4j]'`, or `uv sync` in a checkout). `clm-load-graph` copies the CSV graph into
the database, **replacing what it holds**.

## Documents

Large documents (statutes, policies, contracts) live in a local **collection** with tools to search and read it: you
load the documents and say what they can be filtered on. Searching selects evidence to read; **only your rules
decide**, so no rule should take a search score as an input.

In a branch or decision function, `steps.search(collection, text, filters, limit)`, `steps.document(collection, id,
passage)`, `steps.cites(collection, id, depth)` and `steps.cited_by(collection, id, depth)` give the same rows, each
a traced step. A collection that holds nothing fails the step, naming the command that loads it, rather than giving
no rows that look like an answer.

The settings, the file format, the loading command and what text in documents can do:
[reference](environment-reference.md#documents).

## Decisions

A decision is a DMN decision table, or a function for a rule DMN cannot express:

```toml
[decisions.sla_threshold]
description = "Whether a delivery extension in days is allowed for a key client of the given tier."
dmn = "rules/sla_threshold.dmn"           # its inputs, with their typeRef, are the decision's parameters

[decisions.credit_check]
description = "..."
params = { amount = "number" }
function = "credit_check"                 # credit_check(amount) -> a dict of outputs
```

Each evaluation records its inputs, its outputs and, for a table, the rules that fired. The supported DMN subset and
the error cases: [reference](environment-reference.md#decisions).

## Calculations

```toml
[calculations.branch_figures]
description = "Q3 revenue, cost, tariff cost and margin, lead time, delivery extension and defect rate ..."
params = { q3_units = "number", unit_price = "number", offers = "list", region_b_share = "number" }
function = "branch_figures"               # branch_figures(**inputs) -> a dict of results
```

All arithmetic belongs in a calculation (adding up rows, a margin, a blended lead time), recorded like a rule. What
a calculation may return: [reference](environment-reference.md#calculations).

## Scenarios

```toml
[scenarios.sourcing]
description = "Where to source a product's components: the share bought from Region B. ..."
params = { product = "string" }
levers = { region_b_share = [0, 0.5, 1.0] }     # the only values the agent may choose
branch = "sourcing_branch"
objective = "q3_margin"                         # a key of the figures the branch returns
maximise = true
limits = ["sla_threshold", "quality_threshold", "supply_concentration"]
```

The engine (`engine.py`) runs the branch function once per combination of lever values, or for the branches the agent
asks for. The function sets out a branch's path: which queries run, which rule inputs their rows fill, how the lever
changes them, and which calculation gives the figures. It does everything through `steps` and returns the figures:

```python
def sourcing_branch(steps, params, levers):
    product = steps.query("product_facts", product=params["product"])[0]
    ...
    figures = steps.calculate("branch_figures", q3_units=product["q3_units"], ...)
    steps.decide("quality_threshold", defect_rate=figures["defect_rate"])
    return figures
```

**A branch function moves rows between steps but does no arithmetic.**

Limits, failed branches, ranking and what is a parameter and what a lever:
[reference](environment-reference.md#scenarios).

## Judges

Some evidence is text no rule can read, such as news about a supplier. A judge asks CLM one yes/no question of a text
and gives the probability of yes. That probability is a rule's input, so the rule, not the model, decides.

```toml
[clm]
backend = "mlx"            # or "torch", on Linux with a GPU; model_id and device as in scenario.toml

[judges.delivery_risk]
description = "Whether a piece of news about a supplier is a risk to its deliveries."
instructions = "Does this news mean the supplier may fail to deliver its parts on time in the coming months?"
```

A judgement in a branch, how CLM loads and what the probabilities mean: [reference](environment-reference.md#judges).

## Rules an agent can explore

A decision can carry what a person should know about its rule, and the information it needs. The environment then adds
two tools, `rules` and `rule`, and `describe` lists the topics: the kinds of decision the rules govern. From
[examples/new_site](../examples/new_site/README.md):

```toml
[decisions.stress_test.rule]
statement = "With revenue 20% below plan, a new site must still pay back within 7 years."
reason = "Plans are often optimistic; a site must survive a bad start."
owner = "Finance"
version = "2026-02"
governs = ["open_site"]                  # the topics this rule governs
depends_on = ["payback"]                 # other rules, which must have a rule table too
needs = ["neighbourhood", "household_income", "site_costs"]

[information.neighbourhood]
description = "Which neighbourhood the site would be in (a census tract id, such as tract_4102)"
kind = "supplied"                        # the person asking must give it; no query

[information.household_income]
description = "Median household income in the neighbourhood"
kind = "estimate"                        # or "fact"; both name the query that provides it
query = "household_income"
```

The two tools, citations, what stops loading and facts against estimates:
[reference](environment-reference.md#rules-an-agent-can-explore).

## Risk

```toml
[scenarios.new_site]
...
limits = ["payback", "stress_test", "debt_covenant", "reserve_policy"]
uncertain = ["household_income"]         # estimate queries to try at their low and high
approvals = ["capital_approval"]         # decisions with required, who and reason: listed, never rejecting
```

**Only your rules reject.** A stress test is a limit like any other: in the example, payback with revenue 20% below
plan. Warnings, headroom and approvals: [reference](environment-reference.md#risk).

## The decision record

`clm-agent` writes a decision record beside each answer, as JSON and as a page (`<time>.record.md`). It is built only
from the tools' results, so it works for a server at a URL, and the same calls give the same record. What it holds:
[reference](environment-reference.md#the-decision-record).

## Derived rules and facts

**A rule or fact is either supplied or derived and checked.** Supplied means the owner wrote it: the graph, the DMN
tables, the Python module. Otherwise it can be derived from the environment's own documents by a model and checked
mechanically, and only a person's approval lets it decide. Every decision step, rule and record says
`origin: "supplied"` or `"derived"`. Supplied always decides: derivation never edits a supplied file, and a document
that disagrees with a supplied fact is reported as a conflict. A reading that gave no value, or one a person rejected,
opens no conflict; the conflict of a fact rejected later is dropped at the next `run`.

The environment declares **slots**, the decisions and facts it needs, without their contents; without a `[derived]`
table the layer is ignored:

```toml
[derived]
admit = ["approved"]                 # the default; or ["approved", "checked"], and describe says so
collection = "handbook"              # where the slots are derived from, unless a slot names its own

[derived.decisions.carryover_limit]
description = "How many unused days of paid time off carry into next year, from full years of service."
inputs = { years_of_service = "number" }
outputs = { carryover_limit = "number" }   # a limit adds allowed; reason and citation are always added
owner = "People Operations"
governs = ["benefits"]

[derived.facts.k401_2026.match_cap]        # node k401_2026, property match_cap
description = "The plan's cap on the company match, in dollars a plan year."
type = "number"
```

The commands, the check, the statuses and what loading admits:
[reference](environment-reference.md#derived-rules-and-facts).

### A second reading

The check proves every figure is in the quote; it cannot prove a figure is in the right place ("100% of the first
3%" read as a tier of 100% of pay passes it). A **second reading** by another model looks for that
([reference](environment-reference.md#a-second-reading)).

## The MCP tools

Every tool only reads, and is marked read-only to the client. Each tool's description is yours.

Each tool, its inputs and what it returns: [reference](environment-reference.md#the-mcp-tools).

### Serving settings

An operator can offer fewer tools, in the file or on the command line
([serving.md](serving.md#choosing-the-tools-and-a-read-only-server)):

```toml
[serve]
tools = ["describe", "scenario_*", "search_*", "document_*"]   # only these, and describe always
exclude_tools = ["cites_*", "cited_by_*"]                      # wins over tools
read_only = true                                               # describe says so; an environment's tools only read
```

`clm-server --tools`, `--exclude-tools` and `--read-only` replace the file's setting, never add to it. What is left
out is gone from the tool list, `describe` (which names it under `left_out`), the matching resources and the prompt;
the scenarios still run every part they need. The portal reads the same file, so it honours `[serve]`.

## Resources, a prompt and context

The server also reports its data as MCP resources, its instructions as a prompt, and where each kind of context lives,
so a client or a person can see everything it offers ([serving.md](serving.md#inspecting-a-server) shows how).

The resources, the prompt and where context lives: [reference](environment-reference.md#resources-a-prompt-and-context).

## From a new folder to a registered server

The whole path, with the starter environment:

```bash
uv run clm-new-environment offsite                  # a working environment to edit
npx -y @modelcontextprotocol/inspector@2.13.0 uv run clm-server offsite/environment.toml   # its tools, in the browser
npx -y @modelcontextprotocol/inspector@2.13.0 --cli uv run clm-server offsite/environment.toml \
    --method tools/call --tool-name scenario_offsite --tool-arg team=T1 --tool-arg days=2
```

Expect Leith Studio (V3) best at 6,960 pounds, Northern Works cheaper but rejected as too small for everyone, and Thames
Loft kept at 7,380. Then serve it, list it in a registry and use it:

```bash
export DECISION_SERVER_KEY=<a long random key>
uv run clm-server offsite/environment.toml --http --trace calls.jsonl
uv run clm-registry offsite/environment.toml --name io.github.you/offsite \
    --url https://offsite.corp.internal/mcp --version 0.1.0 --description "Where a team should hold its offsite"
mcp-publisher validate server.json && mcp-publisher publish server.json
npx -y @modelcontextprotocol/inspector@2.13.0 --cli http://127.0.0.1:8000/mcp --transport http \
    --header "Authorization: Bearer <the key>" --method tools/list   # without the header: refused (401)
uv run clm-agent http://127.0.0.1:8000/mcp "Where should the Platform team hold its two-day offsite?"
uv run clm-trace-report calls.jsonl                 # the steps behind each call
```

The registry needs an HTTPS address with a host name others can find. List a server on your own computer under its
future name, and reach it at `127.0.0.1` meanwhile ([registry.md](registry.md#a-private-registry-on-this-computer)).
Change one thing at a time and call it again: a query, a threshold, a lever's values. The Inspector shows what an agent
reads: each branch's figures, rejecting limits and every step.

## The reference agent and the number check

```bash
uv run clm-agent examples/tariff/environment.toml "QUESTION"
uv run clm-agent https://decisions.example.org/mcp "QUESTION"    # a running server, with DECISION_SERVER_KEY
```

It starts `clm-server` for the environment, or connects to one at a URL, and gives a model the server's tools and
nothing else. The default model is Claude through the Anthropic API: Opus 5.5 unless `--model` says otherwise, with the
Anthropic SDK's usual credentials, none stored. The other is a local model through Ollama (`--backend ollama`):
`gemma4:26b` unless `--model` says otherwise, at `--ollama-url` (default `http://127.0.0.1:11434`); Ollama must be
running and the model pulled. Both get the same instructions as the `answer_question` prompt
([reference](environment-reference.md#resources-a-prompt-and-context)): start from `describe`; run the scenario rather
than combine queries and decisions; never calculate; name the rejected branches, with reasons and the dependencies
behind them; cite the steps relied on; say plainly when no tool can answer part of the question.

The question, answer, every call, the model and the time taken go as JSON to the environment's git-ignored `runs/`
(for a URL, `runs/` in the current folder; `clm-new-environment` writes the `.gitignore`), one file per run, with `-2`
for a second run in the same second. The number check then lists any figure in the answer that no tool returned,
reported, not blocked; the citation check does the same for citations.

`--no-tools` and what the checks match: [reference](environment-reference.md#the-number-check).

## What comes later

- **Other graph databases,** such as Memgraph, which also speaks Cypher, through `environment.Connector`.
- **Calibrated judges:** trained on labelled texts (CLM's heads), so a threshold means a probability and not only a
  rank.
