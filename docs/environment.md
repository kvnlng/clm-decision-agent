# Writing an environment

A scenario decides one kind of thing quickly: which route a message takes. An **environment** is for decisions that
rest on an organisation's data, with arithmetic, policy and alternatives to compare, such as "should we move Product
Line X's sourcing to Region B, and what would it do to the Q3 margin and our delivery terms?". Start from
[examples/tariff](../examples/tariff/README.md), which uses everything described here.

## Four parts, and one rule

| Part | Answers | The owner writes |
|---|---|---|
| Knowledge graph | What exists and how it is connected | The data, and named queries over it |
| Decision models | What the rules, formulas and limits say | DMN decision tables, or rule functions |
| Scenarios | Which alternatives there are, and which is best | The levers and their allowed values, the objective and the limits |
| Agent | Turns a question into calls, and the results into an answer | Nothing: any MCP client can be the agent, and `decision_agent.py` is a reference one |

**The owner writes every query, rule, calculation and lever; the agent only chooses among them.** The agent never
writes a graph query, states a rule or invents a lever value, and it is told never to calculate. So every figure and
every verdict in an answer can be traced to the step that produced it, and `number_check.py` checks the answer
against those steps.

Without judges, there is no language model in the server: the same inputs always give the same outputs. A judge
(below) adds one, for evidence no rule can read, and its every answer is traced.

## The file

`environment.toml` sits in a folder with the graph's CSV files, the DMN tables and the owner's Python module. A
mistake stops loading with one line naming the setting.

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

`server.py` serves the folder's decisions when the file has an `[environment]` table, and does not load CLM:

```bash
uv run python server.py examples/tariff/environment.toml            # MCP over stdio
DECISION_SERVER_KEY=... uv run python server.py examples/tariff/environment.toml --http
```

`--http`, the key and `--trace` work as for scenarios ([serving.md](serving.md)); review mode is for CLM scenarios
only.

## The graph

`nodes.csv` has the columns `id,kind,name` and any others; `edges.csv` has `source,relation,target` and any others.
The other columns are properties: an empty cell is no property, and a cell that is a plain decimal (`12`, `-3`,
`0.015`) is a number; `1e5`, `1_000`, `nan` and `inf` stay text. An edge to a node that is not in `nodes.csv` stops
loading, naming the file and line.

The built-in graph (`graph.py`) is held in memory. Named queries are built from its operations:

| Operation | Gives |
|---|---|
| `node(id)`, `nodes(kind)`, `edges(source, relation, target)` | Nodes and edges with their properties |
| `neighbours(id, relation, direction)` | The nodes one edge away: `"out"`, `"in"` or `"both"` |
| `reachable(id, relations, depth)` | The nodes reached following those relations, each with its `path` |
| `single_points_of_failure(ids, relations, depth)` | The nodes every one of `ids` reaches, each with the path from each: losing one would cut off all of them |

## Named queries

```toml
[queries.suppliers_of]
description = "Who supplies a component: region, unit price, lead time in days, defect rate and freight per unit."
params = { component = "string" }         # string, number, integer, boolean or list
function = "suppliers_of"                 # suppliers_of(graph, component) -> a list of dicts
```

The description is what the agent reads, so say what the rows hold and in which units. A question no query can
answer is answered "not in the graph"; the gap shows the owner which query to add. Queries return rows the tool
worked out, not raw data for the agent to sift: a region's name rather than an id to look up.

## On Neo4j

The same named queries can run on a Neo4j database, with the Cypher written beside each query's function:

```toml
[graph]
backend = "neo4j"

[graph.neo4j]
uri = "bolt://127.0.0.1:7687"             # the defaults
user = "neo4j"
password_env = "NEO4J_PASSWORD"           # the variable that holds the password; never the password itself
database = "neo4j"

[queries.suppliers_of]
...
function = "suppliers_of"                 # used on the built-in graph
cypher = """
MATCH (s:Supplier)-[r:SUPPLIES]->(:Node {id: $component})
MATCH (s)-[:LOCATED_IN]->(g:Region)
RETURN s.id AS id, s.name AS name, g.name AS region, r.unit_price AS unit_price, ...
ORDER BY s.id"""
```

- **Each query's parameters** are its Cypher's `$` parameters, and its rows are the `RETURN` columns. Return the
  same columns as the function, so the branch functions work on either graph.
- **On neo4j, every query needs `cypher`;** a query without one stops loading, naming it. Nothing else changes:
  decisions, calculations and scenarios run as before.
- **The driver** is the optional `graph` dependency group: `uv sync --group graph`.
- **The server connects when it starts.** A database it cannot reach, or a password that is wrong, stops start-up
  with one line naming the URI; a query that fails is reported in one line naming the query.

`load_graph.py` copies the environment's CSV graph into the database, **replacing what it holds**:

- each node becomes `(:Node:<Kind> {id, kind, name, ...})`, with its kind capitalised as a label;
- each edge becomes a relationship whose type is the relation in capitals, carrying the edge's other columns;
- `Node.id` is made unique.

```bash
brew install neo4j                        # natively; it brings Java 21
neo4j-admin dbms set-initial-password "$NEO4J_PASSWORD"      # before the first start
neo4j start
uv run --group graph python load_graph.py examples/tariff/environment.toml
```

Keep the password in a git-ignored file such as `.neo4j-local.env` (`NEO4J_PASSWORD=...`) and load it into the
shell with `set -a; . ./.neo4j-local.env; set +a`. With it set, `uv run --group graph pytest -m slow
tests/test_graph_neo4j.py` loads the tariff example into the database and checks that every query and every branch
give the same results on both graphs. Point it only at a database you can wipe.

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

Each evaluation records its inputs, its outputs and, for a table, the rules that fired. The DMN subset supported
(`dmn.py`), read from DMN XML of any version:

- **Hit policies:** `UNIQUE` (the default), `FIRST`, `COLLECT`.
- **Input entries:** `-`; a literal (`1.5`, `"Region B"`, `true`); `<`, `<=`, `>`, `>=`, `=` and `!=` with a number;
  the ranges `[a..b]`, `(a..b)`, `[a..b)` and `(a..b]`; comma-separated lists of these; `not(...)`.
- **Output entries:** number, string and boolean literals.

Anything else is refused when the table loads, naming the file and rule; so are an empty output entry, `not()` and
`not(-)`, and a range whose low end is above its high end. `not(...)` wraps a whole entry, not one item of a list. When no rule matches, or two do under `UNIQUE`, the evaluation is an error, never a silent "allowed"; so is
text given to an entry that tests a number. An input's `typeRef` gives its parameter's type; an input without one
accepts anything, and is still refused by any number test. Input names must be lower-case identifiers, since they
become the tool's arguments. An input entry compares with a literal, so a limit that
differs per client belongs in a column the table can match, such as the client's tier.

## Calculations

```toml
[calculations.branch_figures]
description = "Q3 revenue, cost, tariff cost and margin, lead time, delivery extension and defect rate ..."
params = { q3_units = "number", unit_price = "number", offers = "list", region_b_share = "number" }
function = "branch_figures"               # branch_figures(**inputs) -> a dict of results
```

Arithmetic belongs in a calculation: adding up rows, a margin, a blended lead time. It is recorded like a rule, so
every figure in an answer has a step behind it. Return JSON values: numbers, text, true/false, lists and tables
(dicts). NaN and infinity, `Decimal` and dates do not survive MCP, and an objective that is not a finite number fails
its branch.

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

The engine (`engine.py`) runs the branch function once for each combination of lever values, or for the branches the
agent asks for. The function is the owner's description of a branch's whole path: which queries run, which rule
inputs their rows fill, how the lever changes them, and which calculation gives the figures. It asks for everything
through `steps`, and returns the figures:

```python
def sourcing_branch(steps, params, levers):
    product = steps.query("product_facts", product=params["product"])[0]
    ...
    figures = steps.calculate("branch_figures", q3_units=product["q3_units"], ...)
    steps.decide("quality_threshold", defect_rate=figures["defect_rate"])
    return figures
```

**A branch function moves rows between steps but does no arithmetic of its own.** That is what lets the trace
account for every number.

- **A limit** is a decision whose outputs include `allowed` (true or false) and `reason` (text). A branch is
  **rejected** when any evaluation of a limit says not allowed; the result names the decision, its reason and its
  inputs.
- **A branch fails** when a listed limit was never evaluated or failed (even if the owner's code caught the error),
  when a step fails, or when the figures have no finite number for the objective. So no branch can skip a limit
  unnoticed: in the example, the dependency check runs on every branch because its rule is a limit. A branch that
  has nothing to check a limit against (a product with no key clients, for the SLA rule) fails too, so write the
  branch to check the limit once with a neutral input, or leave the limit out of that scenario.
- **The kept branches are ranked** by the objective. A lever value the owner did not list is refused before anything
  runs.

## Judges

Some evidence is text no rule can read, such as news about a supplier. A judge asks CLM one yes/no question of a
text and gives the probability that the answer is yes. That probability is then a rule's input like any other, so
the rule, not the model, decides.

```toml
[clm]
backend = "mlx"            # or "torch", on Linux with a GPU; model_id and device as for CLM scenarios

[judges.delivery_risk]
description = "Whether a piece of news about a supplier is a risk to its deliveries."
instructions = "Does this news mean the supplier may fail to deliver its parts on time in the coming months?"
```

```python
judged = [(steps.judge("delivery_risk", row["news"])["probability"], row["name"])
          for row in steps.query("supplier_news", suppliers=in_use)]
probability, supplier = max(judged, default=(0.0, "no supplier in use has news"))
steps.decide("news_risk", probability=probability, supplier=supplier)
```

- **CLM loads on the first judgement,** not when the server starts. So an environment, or a scenario, that never
  judges never needs CLM or its model. If CLM cannot load, the judgement fails in one line, its branch fails, and the
  server stays up; the next judgement tries again.
- **Each judgement is a step** in the trace: `{"kind": "judgement", "name", "inputs": {"text"}, "result":
  {"probability"}}`. Empty text is refused, never judged as "no".
- **A judge without `[clm]`** is refused when the environment loads.
- **The probabilities are relative, not calibrated.** An untrained question can lean one way: in the tariff
  example, harmless news scored 0.47 to 0.79 and the risks 0.84 to 0.98. So measure the question on labelled texts
  before choosing a rule's threshold, as `examples/tariff/judge_news.py` does, and say how wide the gap was. Where
  the texts do not separate, the judge is not ready to be a limit.

## The MCP tools

| Tool | Does |
|---|---|
| `query_<name>`, `decision_<name>`, `calculation_<name>` | One named part, with its declared parameters as typed arguments |
| `judge_<name>` | CLM's probability that the judge's question is answered yes for a `text` |
| `scenario_<name>` | Every branch, or the `branches` given: each branch's figures, status, the limits that rejected it and every step |
| `describe` | The environment's name and description, and every part with its parameters, levers, objective and limits |

All of them only read. Each tool's description is the owner's.

## The reference agent and the number check

```bash
uv run --group demo python decision_agent.py examples/tariff/environment.toml "QUESTION"
```

It starts `server.py` for the environment and gives a model the server's tools and nothing else:

- **Claude** through the Anthropic API, the default, with Opus 5.5 unless `--model` says otherwise;
- **a local model** through Ollama: `--backend ollama`, with `gemma4:26b` unless `--model` says otherwise, at
  `--ollama-url` (default `http://127.0.0.1:11434`). Ollama must be running, and the model pulled.

Both get the same instructions and tools. The instructions are to:

- start from `describe`;
- run the scenario rather than combine queries and decisions itself;
- never calculate;
- name the rejected branches, with their reasons and the dependencies behind them;
- cite the steps it relied on;
- say plainly when no tool can answer part of the question.

The question, the answer, every call, the model and the time taken are written as JSON to the environment's `runs/`
folder, which is git-ignored. Claude's credentials are the Anthropic SDK's usual ones; none are stored.

The number check then lists any figure in the answer that no tool returned. A figure matches a tool's number when
the number, in the answer's unit and rounded to the answer's decimals, is the figure: "$1.2 million" matches 1234567,
and "18.4%" matches 0.1843 or 18.43. Halves round up, and signs are not compared, so "fell by 12%" matches -0.12. It
does not check digits inside a word (Q3), dates and times, years, or whole numbers from 0 to 10. A flagged figure is
reported, not blocked.

## What comes later

- **Other graph databases,** such as Memgraph, which also speaks Cypher, through the same connector interface
  (`environment.Connector`).
- **Calibrated judges:** a judge trained on labelled texts (CLM's heads), so its threshold means a probability and
  not only a rank.
