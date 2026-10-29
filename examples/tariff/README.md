# The tariff example: should Product Line X move to Region B?

A COO asks: *"With the recent tariff hikes in Region A, should we shift our primary component sourcing for Product
Line X to Region B, and what would be the impact on our Q3 profit margins and SLA delivery times?"*

This folder is a made-up manufacturer set up as an environment ([docs/environment.md](../../docs/environment.md)),
so that the question can be answered from its graph, its rules and its levers, with every figure traced.

## The company

- **Product Line X** (PX) sells 10,000 units a quarter at $250. Each unit has a housing (plastics), a controller
  board (electronics) and a drive motor (electromechanical).
- **Region A suppliers:** Arden Plastics, Arden Circuits and Aster Motors. Delivery takes 9 to 10 days, the defect
  rate is under 1%, and freight is $4 a unit.
- **Region B suppliers:** Bayline Components (housing and board) and Borealis Motors. Each part costs $1–2 more,
  delivery takes 13 to 14 days, the defect rate is about 1.2%, and freight is $9 a unit.
- **Key clients:** Northwind Hospitals and Corvid Logistics (strategic tier) and Bluepeak Retail (standard tier).
  Harbour Schools buys too, but is not a key client.
- **The planted dependency:** each Region A supplier relies on a different raw-material provider, but both Region B
  suppliers depend on **Kestrel Polymers**, in Region C.

The graph is in `graph/` (20 nodes, 24 edges) and the owner's functions are in `tariff_functions.py`.

## The rules

| Rule | File | Says |
|---|---|---|
| Tariff rate | `rules/tariff_rate.dmn` | Region A after the hike: 25% on plastics, 30% on electronics, 25% on electromechanical. Region B: 5%, 8%, 6% |
| SLA threshold | `rules/sla_threshold.dmn` | Deliveries may slip by at most 5 days for a strategic client, 10 for a standard one |
| Quality threshold | `rules/quality_threshold.dmn` | Product Line X allows a defect rate of at most 1.5% |
| Supply concentration | `rules/supply_concentration.dmn` | No raw-material provider may carry more than 60% of supply |

The last three are the `sourcing` scenario's limits, so every branch checks them all: the SLA rule once per key
client. A product with no key clients would fail every branch, because the SLA limit is then never checked; that is
deliberate. An owner who wants such products compared would check the rule once with a neutral input, or leave the
limit out of the scenario.

## The levers

The share of each component bought from Region B: 0, 50% or 100%. The objective is the Q3 margin. A branch's lead
time is each component's volume-weighted lead time, and the product waits for its slowest component. The delivery
extension is measured against buying everything in Region A.

| Region B share | Q3 margin | Delivery extension | Defect rate | Largest share on one provider | Result |
|---|---|---|---|---|---|
| 0 | $662,500 | 0 days | 0.80% | 33% | kept |
| 50% | $704,900 | 2 days | 1.02% | 50%, Kestrel Polymers | **kept, best** |
| 100% | $747,300 | 4 days | 1.23% | 100%, Kestrel Polymers | rejected: supply concentration |

Moving everything would earn the most, and it meets the delivery and quality limits. It is rejected because all
supply would then depend on Kestrel Polymers. Half is the best branch no limit rejects.

## Run it

```bash
uv run python server.py examples/tariff/environment.toml     # serve it over MCP, for any agent
uv run --group demo python decision_agent.py examples/tariff/environment.toml "With the recent tariff hikes in Region A, should we shift our primary component sourcing for Product Line X to Region B, and what would be the impact on our Q3 profit margins and SLA delivery times?"
uv run --group demo python decision_agent.py examples/tariff/environment.toml "..." --backend ollama   # gemma4:26b
```

On Neo4j: load the graph, then set `[graph] backend = "neo4j"` in `environment.toml`. Each query's Cypher is beside
its function, and `tests/test_graph_neo4j.py -m slow` checks that both graphs give the same rows and branches
([docs/environment.md](../../docs/environment.md#on-neo4j)).

```bash
set -a; . ./.neo4j-local.env; set +a
uv run --group graph python load_graph.py examples/tariff/environment.toml
```

## What happened when the models were asked

On 29 October 2026, the question above went to Claude Opus 5.5 twice and to `gemma4:26b` three times, running
locally through Ollama on an M5 Max. All the runs used the built-in graph.

| Run | Tool calls | Recommendation | Named Kestrel Polymers | Said both Region B suppliers depend on it | Figures no tool returned | Time |
|---|---|---|---|---|---|---|
| Claude, 1 | `describe`, `scenario_sourcing` | half to Region B | yes | yes | none | not recorded |
| Claude, 2 | `describe`, `scenario_sourcing` | half to Region B | yes | yes | **$42,400** | 17.8 s |
| gemma4, 1 | `describe`, `scenario_sourcing` | half to Region B | yes | no | none | 20.8 s |
| gemma4, 2 | `describe`, `scenario_sourcing` | half to Region B | yes | no | none | 12.1 s |
| gemma4, 3 | `describe`, `scenario_sourcing` | half to Region B | yes | no | none | 19.0 s |

- **Every run reached the same decision, by the same two calls.** That is the design working: the scenario did the
  deciding, and the models reported it.
- **Claude's second answer broke the rule against calculating.** It gave the gain of moving half as "$42,400". That
  is the difference between two figures the tools returned, worked out by the model, and the number check flagged
  it. The arithmetic happens to be right, but the figure came from the model, not from any traced step.
- **Claude's answers were fuller, and explained the dependency.** They listed:
  - the tariff rates;
  - the three key clients by name and tier;
  - the margin rate and tariff cost of every branch;
  - that both Region B suppliers buy from Kestrel Polymers.

  Both said what the analysis leaves out. The first named shares between 50% and 60%, and transition time; the
  second named switching costs and supplier capacity.
- **gemma4's answers were shorter, and never did arithmetic of their own.** They gave:
  - the margin and the delivery extension;
  - in one of the three, the defect rate;
  - the rejection with its reason, naming Kestrel Polymers.

  None said why Kestrel carries all the supply (both Region B suppliers depend on it), although the instructions
  ask for the dependency behind a rejection. None mentioned what was not modelled. They printed figures as the
  tools gave them ("704,900.0").
- **gemma4 was free and stayed on the Mac.** Claude was about as fast, at the cost of an API call.
- **On Neo4j, gemma4 made one labelling mistake.** In a sixth run, with the example on Neo4j and a shorter question,
  it made the same two calls, took 14.6 s and recommended half. But it headed the 0% branch "Rejected", although it
  passed every limit and only ranked lower. The number check cannot catch that: it checks figures, not words.

Six runs on a small, made-up graph show the pattern works with both models. They don't measure how often an agent
gets it right. A fair next measure would be many varied questions, each scored for whether the right branch was
recommended, whether the rejected branch and its dependency were named, and how many figures were flagged.
