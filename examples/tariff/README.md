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
| News risk | `rules/news_risk.dmn` | No supplier in use may have news CLM judges a delivery risk with probability above 0.82 |

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

## The news

Three suppliers have news on the graph (the `news` column of `graph/nodes.csv`). Borealis Motors, Region B's motor
supplier: workers at its only plant have voted to strike from 1 August. Bayline Components has opened a second
line, and Arden Plastics was named supplier of the year. No rule can read that, so the `sourcing_with_news` scenario
asks CLM, through the `delivery_risk` judge, whether each news line about a supplier in use is a delivery risk. The
highest probability goes to the `news_risk` limit. The `sourcing` scenario is unchanged, and never loads CLM.

| Region B share | News judged (CLM's probability of a delivery risk) | Result |
|---|---|---|
| 0 | Arden Plastics' award 0.792 | **kept, best** |
| 50% | Arden 0.792, Bayline's new line 0.772, Borealis' strike 0.921 | rejected: news risk |
| 100% | Bayline 0.772, Borealis 0.921 | rejected: supply concentration and news risk |

**How the threshold was chosen.** `judge_news.py` asks the same question of twelve made-up, labelled news lines
(`news_cases.jsonl`) and prints each probability, on MLX on the Mac:

| Probability | Delivery risk? | News |
|---|---|---|
| 0.975 | yes | Coriander Metals has filed for insolvency |
| 0.950 | yes | Flooding has closed Bayline's port |
| 0.921 | yes | Borealis Motors' workers have voted to strike |
| 0.919 | yes | A cyberattack has shut Altamira Resins' ordering systems |
| 0.904 | yes | Region A has banned exports of electric motors |
| 0.843 | yes | A fire destroyed Halden Steel's rolling mill |
| 0.792 | no | Arden Plastics was named supplier of the year |
| 0.772 | no | Bayline opened a second assembly line |
| 0.758 | no | Borealis published unchanged prices |
| 0.715 | no | Arden Circuits appointed a new chief executive |
| 0.656 | no | Aster Motors moved its head office |
| 0.470 | no | Kestrel Polymers renewed its supply contracts |

The two groups separate, so 0.82, between 0.792 and 0.843, rejects every risk and none of the others. **But the gap
is only 0.05, and the question leans towards yes:** an award scored 0.79. CLM's probabilities are relative, not
calibrated, and twelve lines are a small sample. A milder risk, or harmless news worded more alarmingly, could land
on the wrong side. The judgement is a reason for a person to look at Borealis Motors, not a verdict to act on alone.

## Run it

```bash
uv run python server.py examples/tariff/environment.toml     # serve it over MCP, for any agent
uv run --group demo python decision_agent.py examples/tariff/environment.toml "With the recent tariff hikes in Region A, should we shift our primary component sourcing for Product Line X to Region B, and what would be the impact on our Q3 profit margins and SLA delivery times?"
uv run --group demo python decision_agent.py examples/tariff/environment.toml "..." --backend ollama   # gemma4:26b
uv run python examples/tariff/judge_news.py   # the news measurement, with CLM
```

On Neo4j: load the graph, then set `[graph] backend = "neo4j"` in `environment.toml`. Each query's Cypher is beside
its function, and `tests/test_graph_neo4j.py -m slow` checks that both graphs give the same rows and branches
([docs/environment.md](../../docs/environment.md#on-neo4j)). Reload the graph whenever `graph/` changes: a database
loaded before the `news` column has no news, so `sourcing_with_news` would find none and decide the news limit with
0.0 (its trace then says "no supplier in use has news").

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

**With the news, the two models gave different answers.** The same question went to each model once more, with the
server able to judge news. Neither was told about the news.

| Run | Tool calls | Recommendation | Mentioned the strike | Figures no tool returned | Time |
|---|---|---|---|---|---|
| Claude | `describe`, `scenario_sourcing_with_news`, `scenario_sourcing` | stay in Region A | yes | none | 30.0 s |
| gemma4 | `describe`, `scenario_sourcing` | half to Region B | no | none | 15.2 s |

- **Claude chose the news scenario and recommended staying in Region A.** It said why half was rejected (the strike,
  judged 0.92), and why everything was rejected (Kestrel Polymers and the strike). It also ran `sourcing` to show
  half would be best without the news. Unprompted, it warned that CLM also scored the award 0.79 and the new line
  0.77, so the rejection rests on a narrow gap and a person should check the strike. It said the model cannot test
  a mix such as keeping motors in Region A.
- **gemma4 ran the scenario without the news and recommended half.** It was twice as fast and its figures were all
  traced, but it did not run the scenario that checks the news, so it missed the strike. The number check cannot
  catch that: every figure was right, for a question without the news.
- So the framework gives the agent the evidence; choosing to look at it is still the agent's. An owner who wants the
  news always checked should offer one scenario, with the news limit, rather than two.

Eight runs on a small, made-up graph show the pattern works with both models. They don't measure how often an agent
gets it right. A fair next measure would be many varied questions, each scored for whether the right branch was
recommended, whether the rejected branch and its dependency were named, and how many figures were flagged.
