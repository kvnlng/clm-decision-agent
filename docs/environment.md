# Writing an environment

Your environment is the tools you define: queries that reach your data, actions, and your rules. A decision runs over
it in one of two styles: routing a message with CLM ([scenario.md](scenario.md)), or weighing options against your
rules, which this file, `environment.toml`, describes. Weighing suits decisions that rest on an organisation's data,
with arithmetic, policy and alternatives to compare: "should we move Product Line X's sourcing to Region B, and what
would it do to the Q3 margin and our delivery terms?".

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
graph query, states a rule or invents a lever value, and is told never to calculate. So every figure and verdict traces
to a step, and `number_check.py` checks the answer against those steps.

Without judges, the server has no language model: the same inputs give the same outputs. A judge (below) adds one, for
evidence no rule can read, and every answer it gives is traced.

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

`clm-server` serves a file with an `[environment]` table without loading CLM:

```bash
uv run clm-server examples/tariff/environment.toml            # MCP over stdio
DECISION_SERVER_KEY=... uv run clm-server examples/tariff/environment.toml --http
```

`clm-call` starts the server and lists its tools, or calls one, with no agent:

```bash
uv run clm-call examples/tariff/environment.toml
uv run clm-call examples/tariff/environment.toml scenario_sourcing '{"product": "PX"}'
uv run clm-call http://127.0.0.1:8000/mcp describe     # a running server, with the key in DECISION_SERVER_KEY
```

`--http`, the key and `--trace` work as in routing ([serving.md](serving.md)). Review mode is for routing only.

## The graph

`nodes.csv` has the columns `id,kind,name` and any others; `edges.csv` has `source,relation,target` and any others.
Extra columns are properties. An empty cell is no property; a plain decimal (`12`, `-3`, `0.015`) is a number; `1e5`,
`1_000`, `nan` and `inf` stay text. An edge to an unknown node stops loading, naming the file and line.

Build named queries from the operations of the in-memory graph (`graph.py`):

| Operation | Gives |
|---|---|
| `node(id)`, `nodes(kind)`, `edges(source, relation, target)` | Nodes and edges with their properties |
| `neighbours(id, relation, direction)` | The nodes one edge away: `"out"`, `"in"` or `"both"` |
| `reachable(id, relations, depth)` | The nodes reached following those relations, each with its `path` |
| `single_points_of_failure(ids, relations, depth)` | The nodes every one of `ids` reaches, with the path from each: losing one cuts off all of them |

## Named queries

```toml
[queries.suppliers_of]
description = "Who supplies a component: region, unit price, lead time in days, defect rate and freight per unit."
params = { component = "string" }         # string, number, integer, boolean or list
function = "suppliers_of"                 # suppliers_of(graph, component) -> a list of dicts
```

**A query must fail when what it was asked about does not exist.** No rows looks like an answer: in the offsite
example, an unknown team had nobody in it, so every venue fitted and the cheapest won. Raise an error
(`LookupError(f"no team {team}")`) and the branch fails, naming it.

The agent reads the description, so say what the rows hold and in which units. A question no query can answer gets "not
in the graph", which shows which query to add. Return worked-out rows, not raw data: a region's name, not an id to look
up.

## On Neo4j

Named queries can run on Neo4j, with the Cypher beside each query's function:

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

- **Parameters and rows:** a query's parameters are its Cypher's `$` parameters, its rows the `RETURN` columns. Return
  the same columns as the function, so branch functions work on either graph.
- **Every query needs `cypher`;** one without it stops loading, naming it. Nothing else changes.
- **The driver** is the `neo4j` extra: `pip install 'clm-decision-agent[neo4j]'`, or `uv sync` in a checkout.
- **The server connects at start-up.** An unreachable database or wrong password stops it with one line naming the URI;
  a failed query is reported in one line naming the query.

`clm-load-graph` copies the environment's CSV graph into the database, **replacing what it holds**:

- each node becomes `(:Node:<Kind> {id, kind, name, ...})`, its kind capitalised as a label;
- each edge becomes a relationship typed by the relation in capitals, carrying the edge's other columns;
- `Node.id` is made unique.

```bash
brew install neo4j                        # natively; it brings Java 21
neo4j-admin dbms set-initial-password "$NEO4J_PASSWORD"      # before the first start
neo4j start
uv run clm-load-graph examples/tariff/environment.toml
```

Keep the password in a git-ignored file such as `.neo4j-local.env` (`NEO4J_PASSWORD=...`) and load it with
`set -a; . ./.neo4j-local.env; set +a`. `uv run pytest -m slow tests/test_graph_neo4j.py` then loads the tariff example
and checks that every query and branch match on both graphs. Point it only at a database you can wipe.

## Documents

Some decisions rest on large documents: statutes and court opinions, policies, manuals, contracts. The framework keeps
them in a local **collection** and gives the agent tools to search and read it, so you write no tool code: you load
the documents and say what they can be filtered on. Searching selects evidence to read; **only your rules decide**,
so no rule should take a search score as an input.

```toml
[documents]
backend = "postgres"                      # the defaults
host = "127.0.0.1"
port = 5432
database = "clm_documents"
user = "clm"
password_env = "CLM_DOCUMENTS_PASSWORD"   # the variable that holds the password; never the password itself
ollama_url = "http://127.0.0.1:11434"     # embeds passages and searches, so no text leaves the machine
tools = true                              # false: only your functions search; the agent gets no document tools
cited_text = true                         # false: a rule's citations give the document's citation and title, not its text

[collections.cases]
description = "Court opinions on non-competes: what the agent reads about the collection."
embedding_model = "embeddinggemma"        # an Ollama embedding model
filters = { state = "string", court_level = "string", year = "integer", decided = "date" }
```

- **Filter types** are `string`, `integer`, `number`, `boolean` and `date` (ISO, `2024-05-01`). A search filters on a
  value, a list of values (any of them), or `{"from", "to"}` for numbers and dates. An undeclared field or a value of
  the wrong type is refused in one line.
- **Passages** are found by meaning (pgvector) and by words (PostgreSQL full-text search), the two rankings combined.
  Each comes with its document id, citation, title, metadata, passage label and score.
- **The server connects at start-up.** An unreachable database, or one without the `vector` extension, stops it with
  one line. The driver is the `documents` extra: `pip install 'clm-decision-agent[documents]'`.

`clm-load-documents` loads a collection from JSON lines, **replacing what it holds**:

```json
{"id": "tx-bcc-15.50", "citation": "Tex. Bus. & Com. Code § 15.50", "title": "Criteria for enforceability",
 "metadata": {"state": "TX"}, "passages": [{"label": "(a)", "text": "..."}, {"label": "(b)", "text": "..."}]}
{"id": "cl-4321", "citation": "100 S.W.3d 1", "title": "...", "metadata": {"state": "TX", "year": 2019},
 "text": "The opinion's text ...", "cites": [{"id": "tx-bcc-15.50", "passage": "3"}]}
```

- `passages` keeps your own split, such as a statute's subsections; `text` is split into passages for you.
- `metadata` holds only the declared filters, plus `sources`: which passage each value came from.
- `cites` lists the ids a document cites, as text or with the citing passage. An id outside the file is kept and
  counted, since documents cite others you may not hold.
- The collection records when it was built, from what and with which model. Every search reports that date, and
  searching with a different model is refused.

```bash
brew install postgresql@18 pgvector       # natively
createdb clm_documents
psql -d clm_documents -c "CREATE ROLE clm LOGIN PASSWORD '...'; ALTER DATABASE clm_documents OWNER TO clm"
ollama pull embeddinggemma
set -a; . ./.postgres-local.env; set +a   # CLM_DOCUMENTS_PASSWORD=...
uv run clm-load-documents my_environment/environment.toml cases cases.jsonl --source "CourtListener, 2026-11"
```

In a branch or decision function, `steps.search(collection, text, filters, limit)`, `steps.document(collection, id,
passage)`, `steps.cites(collection, id, depth)` and `steps.cited_by(collection, id, depth)` give the same rows, each
a traced step. A collection that holds nothing fails the step, naming the command that loads it, rather than giving
no rows that look like an answer. `uv run pytest -m postgres` checks the store against a local database
`clm_documents_test`, with its password in `CLM_TEST_POSTGRES_PASSWORD`; point it only at a database you can wipe.

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

Each evaluation records its inputs, its outputs and, for a table, the rules that fired. The supported DMN subset
(`dmn.py`), read from DMN XML of any version:

- **Hit policies:** `UNIQUE` (the default), `FIRST`, `COLLECT`.
- **Input entries:** `-`; a literal (`1.5`, `"Region B"`, `true`); `<`, `<=`, `>`, `>=`, `=` and `!=` with a number;
  the ranges `[a..b]`, `(a..b)`, `[a..b)` and `(a..b]`; comma-separated lists of these; `not(...)`, which wraps a whole
  entry, not one item of a list.
- **Output entries:** number, string and boolean literals.

Refused at load, naming the file and rule: anything else, an empty output entry, `not()`, `not(-)`, and a range whose
low end is above its high end.

An evaluation is an error, never a silent "allowed", when no rule matches, two match under `UNIQUE`, or text reaches an
entry that tests a number.

An input's `typeRef` gives its parameter's type; without one it accepts anything, but a number test refuses non-numbers.
Input names must be lower-case identifiers, as they become the tool's arguments. Entries compare with literals, so a
limit that differs per client belongs in a column the table can match, such as the client's tier.

## Calculations

```toml
[calculations.branch_figures]
description = "Q3 revenue, cost, tariff cost and margin, lead time, delivery extension and defect rate ..."
params = { q3_units = "number", unit_price = "number", offers = "list", region_b_share = "number" }
function = "branch_figures"               # branch_figures(**inputs) -> a dict of results
```

All arithmetic belongs in a calculation: adding up rows, a margin, a blended lead time. It is recorded like a rule, so
every figure has a step behind it. Return JSON values: numbers, text, true/false, lists and dicts. NaN, infinity,
`Decimal` and dates do not survive MCP, and an objective that is not a finite number fails its branch.

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

**A branch function moves rows between steps but does no arithmetic,** so the trace accounts for every number.

- **A limit** is a decision whose outputs include `allowed` (true or false) and `reason` (text). A branch is
  **rejected** when any evaluation of a limit says not allowed; the result names the decision, reason and inputs.
- **A branch fails** when a listed limit was never evaluated or failed (even if your code caught the error), a step
  fails, or the figures have no finite number for the objective. So no branch skips a limit: in the example, the
  dependency check runs on every branch because its rule is a limit. A branch with nothing to check a limit against (a
  product with no key clients, for the SLA rule) fails too: check the limit once with a neutral input, or leave it out
  of that scenario.
- **Kept branches are ranked** by the objective. An unlisted lever value is refused before anything runs.

**What the question fixes is a parameter; only real alternatives are levers.** In the offsite example the team and the
number of days are parameters, and the venue is the lever. With days as a lever, ranking by lowest cost would always
choose one day, though the asker had already fixed the length.

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

```python
judged = [(steps.judge("delivery_risk", row["news"])["probability"], row["name"])
          for row in steps.query("supplier_news", suppliers=in_use)]
probability, supplier = max(judged, default=(0.0, "no supplier in use has news"))
steps.decide("news_risk", probability=probability, supplier=supplier)
```

- **CLM loads on the first judgement,** not at start-up, so what never judges never needs CLM or its model. If CLM
  cannot load, the judgement fails in one line, its branch fails and the server stays up; the next judgement retries.
- **Each judgement is a step** in the trace: `{"kind": "judgement", "name", "inputs": {"text"}, "result":
  {"probability"}}`. Empty text is refused, never judged as "no".
- **A judge without `[clm]`** is refused at load.
- **Probabilities are relative, not calibrated.** An untrained question can lean one way: in the tariff example,
  harmless news scored 0.47 to 0.79 and risks 0.84 to 0.98. Measure the question on labelled texts before choosing a
  threshold, as `examples/tariff/judge_news.py` does, and say how wide the gap was. If the texts do not separate, the
  judge is not ready to be a limit.

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

- **`rules(topic)`** gives the rules governing the topic and those they depend on, dependencies first, each with
  statement, reason, owner and version; the information they need; and in `to_ask`, what the asker must supply.
- **`rule(name)`** gives one rule, with its parameters and, for a DMN table, its rows as written.
- **Name a supplied item after the scenario parameter that carries it,** so the decision record can tell what was given
  from what is still to ask.
- **A rule can cite its source** in a collection (see Documents): `cites = ["tx-bcc-15.50"]` in its rule table, and
  `collection = "laws"` on the decision. A decision table can carry a `citation` output, ids separated by `;`, so each
  row says where it comes from. `rule` and `rules` show them. An id the collection does not hold stops loading, so no
  rule cites what cannot be read.
- **These stop loading, naming the mistake:** `needs` that are not `[information.*]` items, `depends_on` that are not
  rules, a rule depending on itself through others, and an information item whose kind differs from its query's.

## Facts, estimates and what was supplied

A query is a **fact** unless it says `kind = "estimate"`. Each estimate row has `low`, `central`, `high`, `source` and
`as_of`; a row without them, or with low above central or central above high, fails the step in one line. The engine
adds `value`: the central value, or the low or high in a risk run, so the branch reads `row["value"]` and never chooses.
Every query step records `"information": "fact"` or `"estimate"`. A scenario's result says what was supplied (its
parameters), the rules applied and the information used.

## Risk

```toml
[scenarios.new_site]
...
limits = ["payback", "stress_test", "debt_covenant", "reserve_policy"]
uncertain = ["household_income"]         # estimate queries to try at their low and high
approvals = ["capital_approval"]         # decisions with required, who and reason: listed, never rejecting
```

- **Only your rules reject.** A stress test is a limit like any other: in the example, payback with revenue 20% below
  plan.
- **Warnings.** Each kept branch reruns at each uncertain estimate's low and high. A limit that fails there is a warning
  on the branch (the estimate, case, limit and reason). The branch stays kept, and ranking uses the central case.
- **Headroom.** For each kept branch and uncertain estimate, how far the estimate can move from central before a limit
  fails: "income can fall 2% before the stress test fails". Found by bisection to within half a percent, over a fall to
  zero or a rise to double. It assumes a limit's verdict changes once, as for a threshold.
- **Approvals.** A decision in `approvals` outputs `required` (true or false), `who` and `reason`. A branch that needs
  one lists it, so "the board must approve" is not mistaken for a rejection. An approval never checked fails the branch,
  like a limit.

## The decision record

`clm-agent` writes a decision record beside each answer, as JSON and as a page (`<time>.record.md`). It is built only
from the tools' results, so it works for a server at a URL, and the same calls give the same record. It holds the
question; what was supplied; the rules applied, with statement, reason, owner and version; every fact with its values
and every estimate with its range, source and date; each scenario's branches with figures, rejecting limits, warnings,
headroom and approvals; what is still to ask; and any failed call. Then comes "The agent's explanation (not checked)",
with any figure the number check could not find. Figures in the question count as given.

When the rules need something the question does not give, the agent is told to stop and list it, with the rule that
needs it, not run a scenario or fill it in.

## The MCP tools

| Tool | Does |
|---|---|
| `query_<name>`, `decision_<name>`, `calculation_<name>` | One named part, its declared parameters as typed arguments |
| `judge_<name>` | CLM's probability that the judge's question is answered yes for a `text` |
| `search_<name>`, `document_<name>` | A collection's closest passages to a `text`, with `filters` and `limit`; one document or passage as stored |
| `cites_<name>`, `cited_by_<name>` | The documents one cites, or that cite it with the citing passage, to a `depth` |
| `scenario_<name>` | Every branch, or the `branches` given: figures, status, rejecting limits and every step |
| `describe` | Name, description, and every part with its parameters, levers, objective and limits |
| `rules`, `rule` | With rule metadata: the rules governing a topic, and one rule with its rows (above) |

All only read. Each tool's description is yours.

## From a new folder to a registered server

The whole path, with the starter environment.

```bash
uv run clm-new-environment offsite                  # a working environment to edit
uv run clm-call offsite/environment.toml            # its tools, without an agent
uv run clm-call offsite/environment.toml scenario_offsite '{"team": "T1", "days": 2}'
```

Expect Leith Studio (V3) best at 6,960 pounds, Northern Works cheaper but rejected as too small for everyone, and Thames
Loft kept at 7,380. Then serve it, list it in a registry and use it:

```bash
export DECISION_SERVER_KEY=<a long random key>
uv run clm-server offsite/environment.toml --http --trace calls.jsonl
uv run clm-registry offsite/environment.toml --name io.github.you/offsite \
    --url https://offsite.corp.internal/mcp --version 0.1.0 --description "Where a team should hold its offsite"
mcp-publisher validate server.json && mcp-publisher publish server.json
uv run clm-call http://127.0.0.1:8000/mcp describe  # without the key: refused (401)
uv run clm-agent http://127.0.0.1:8000/mcp "Where should the Platform team hold its two-day offsite?"
uv run clm-trace-report calls.jsonl                 # the steps behind each call
```

- **The registry needs an HTTPS address with a host name** others can find. List a server on your own computer under
  its future name, and reach it at `127.0.0.1` meanwhile
  ([registry.md](registry.md#a-private-registry-on-this-computer)).
- **Change one thing at a time and call it again:** a query, a threshold, a lever's values. `clm-call` shows what an
  agent reads: each branch's figures, rejecting limits and every step.

## The reference agent and the number check

```bash
uv run clm-agent examples/tariff/environment.toml "QUESTION"
uv run clm-agent https://decisions.example.org/mcp "QUESTION"    # a running server, with DECISION_SERVER_KEY
```

It starts `clm-server` for the environment, or connects to one at a URL, and gives a model the server's tools and
nothing else:

- **Claude** through the Anthropic API, the default, with Opus 5.5 unless `--model` says otherwise;
- **a local model** through Ollama: `--backend ollama`, with `gemma4:26b` unless `--model` says otherwise, at
  `--ollama-url` (default `http://127.0.0.1:11434`). Ollama must be running, and the model pulled.

Both get the same tools and instructions: start from `describe`; run the scenario rather than combine queries and
decisions; never calculate; name the rejected branches, with reasons and the dependencies behind them; cite the steps
relied on; say plainly when no tool can answer part of the question.

The question, answer, every call, the model and the time taken go as JSON to the environment's git-ignored `runs/`
(for a URL, `runs/` in the current folder). Claude's credentials are the Anthropic SDK's usual ones; none are stored.

The number check then lists any figure in the answer that no tool returned. A figure matches a tool's number when that
number, in the answer's unit and rounded to its decimals, is the figure: "$1.2 million" matches 1234567, and "18.4%"
matches 0.1843 or 18.43. Halves round up and signs are ignored, so "fell by 12%" matches -0.12. It skips digits inside
a word (Q3), dates and times, years, and whole numbers from 0 to 10. A flagged figure is reported, not blocked.

The citation check does the same for citations, so an answer cannot cite a case or section no tool returned:

- **A statute** matches by its section: "§15.50", "§ 15.50" and "Tex. Bus. & Com. Code Ann. § 15.50" all match a
  stored "Tex. Bus. & Com. Code § 15.50". A subsection, "§15.51(c)", matches when the tools returned that section's
  passage labelled "(c)", or a citation naming it. "Section 15.50", "§§ 15.50 and 15.51", "820 ILCS 90/10" and
  "RCW 49.62.020" are read too. Sections are compared without their code.
- **A reporter citation**, "100 S.W.3d 1", matches by volume, reporter and page.
- **A case name with no citation after it**, "Smith v. Jones", cannot be checked: it is listed, never passed.

The decision record lists each citation found with the stored passage it matches, then those no tool returned and
those not checked. A decision table's `citation` ids are given with each document's citation and title in its step.

## What comes later

- **Other graph databases,** such as Memgraph, which also speaks Cypher, through `environment.Connector`.
- **Calibrated judges:** trained on labelled texts (CLM's heads), so a threshold means a probability and not only a rank.
