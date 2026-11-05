# The tariff example: should Product Line X move to Region B?

A COO asks: *"With the recent tariff hikes in Region A, should we shift our primary component sourcing for Product
Line X to Region B, and what would be the impact on our Q3 profit margins and SLA delivery times?"*

The folder is a made-up manufacturer set up as an environment ([docs/environment.md](../../docs/environment.md)):
the answer comes from its graph, rules and levers, every figure traced.

## The company

`graph/nodes.csv` and `graph/edges.csv` (20 nodes, 24 edges) hold every supplier's prices, lead times, defect rates
and freight; the owner's functions are in `tariff_functions.py`. What the findings depend on:

- Product Line X (`PX`): 10,000 units a quarter at $250; a housing (plastics), a controller board (electronics)
  and a drive motor (electromechanical).
- Region A suppliers (Arden Plastics, Arden Circuits, Aster Motors): 9 to 10 days, under 1% defects. Region B
  (Bayline Components, Borealis Motors): $1–2 more a part, 13 to 14 days, about 1.2% defects, far less tariff.
- Key clients: Northwind Hospitals and Corvid Logistics (strategic), Bluepeak Retail (standard); Harbour Schools
  is not key.
- The planted dependency: each Region A supplier has its own raw-material provider; both Region B suppliers depend
  on **Kestrel Polymers** in Region C.

## The rules

| Rule (`rules/*.dmn`) | Threshold |
|---|---|
| `tariff_rate` | Region A after the hike: 25% plastics, 30% electronics, 25% electromechanical. Region B: 5%, 8%, 6% |
| `sla_threshold` | Deliveries may slip at most 5 days for a strategic client, 10 for a standard one |
| `quality_threshold` | Defect rate at most 1.5% |
| `supply_concentration` | No raw-material provider may carry more than 60% of supply |
| `news_risk` | No supplier in use may have news that CLM judges a delivery risk with probability above 0.82 |

The `sourcing` scenario's limits are the SLA, quality and concentration rules, checked by every branch, the SLA rule
once per key client. A product with no key clients fails every branch, since the SLA limit is never checked; that
is deliberate. To compare such products, check the rule once with a neutral input or leave the limit out.

## The levers

The lever is the share of each component bought from Region B: 0, 50% or 100%; the objective is the Q3 margin.
Lead time is volume-weighted per component and the product waits for its slowest; the extension is measured against
buying everything in Region A. A component with no offer in a region the branch buys from, or a supplier outside
Regions A and B, stops the branch with an error naming it.

| Region B share | Q3 margin | Extension | Defect rate | Largest share on one provider | Result |
|---|---|---|---|---|---|
| 0 | $662,500 | 0 days | 0.80% | 33% | kept |
| 50% | $704,900 | 2 days | 1.02% | 50%, Kestrel Polymers | **kept, best** |
| 100% | $747,300 | 4 days | 1.23% | 100%, Kestrel Polymers | rejected: supply concentration |

Moving everything earns the most and meets the delivery and quality limits, but all supply would depend on Kestrel
Polymers. Half is the best branch no limit rejects.

## The news

Three suppliers have news (the `news` column of `graph/nodes.csv`): a strike at Borealis Motors' only plant from
1 August, a second line at Bayline, an award for Arden Plastics. No rule can read that, so `sourcing_with_news`
asks CLM through the `delivery_risk` judge whether each news line about a supplier in use is a delivery risk and
passes the highest probability to `news_risk`. `sourcing` is unchanged and never loads CLM.

| Region B share | News judged (CLM's probability of a delivery risk) | Result |
|---|---|---|
| 0 | Arden's award 0.792 | **kept, best** |
| 50% | Arden 0.792, Bayline's line 0.772, Borealis' strike 0.921 | rejected: news risk |
| 100% | Bayline 0.772, Borealis 0.921 | rejected: supply concentration, news risk |

**How the threshold was chosen.** `judge_news.py` asks the same question of twelve made-up, labelled lines
(`news_cases.jsonl`). On MLX on the Mac:

| Probability | Risk? | News |
|---|---|---|
| 0.975 | yes | Coriander Metals insolvent |
| 0.950 | yes | Flooding closed Bayline's port |
| 0.921 | yes | Borealis Motors' strike vote |
| 0.919 | yes | Cyberattack shut Altamira Resins' ordering |
| 0.904 | yes | Region A banned motor exports |
| 0.843 | yes | Fire destroyed Halden Steel's mill |
| 0.792 | no | Arden Plastics supplier of the year |
| 0.772 | no | Bayline opened a second line |
| 0.758 | no | Borealis' prices unchanged |
| 0.715 | no | Arden Circuits' new chief executive |
| 0.656 | no | Aster Motors moved head office |
| 0.470 | no | Kestrel Polymers renewed contracts |

The groups separate, so 0.82 rejects every risk and none of the others. But the gap is only 0.05 and the question
leans towards yes (an award scored 0.79); CLM's probabilities are relative, not calibrated, and twelve lines are a
small sample. The judgement is a reason for a person to look at Borealis Motors, not a verdict to act on alone.

## Run it

```bash
uv run clm-server examples/tariff/environment.toml     # serve it over MCP, for any agent
uv run clm-agent examples/tariff/environment.toml "With the recent tariff hikes in Region A, should we shift our primary component sourcing for Product Line X to Region B, and what would be the impact on our Q3 profit margins and SLA delivery times?"
uv run clm-agent examples/tariff/environment.toml "..." --backend ollama   # gemma4:26b
uv run python examples/tariff/judge_news.py   # the news measurement, with CLM
```

On Neo4j: load the graph, then set `[graph] backend = "neo4j"` in `environment.toml`. Each query's Cypher is beside
its function; `tests/test_graph_neo4j.py -m slow` checks both graphs give the same rows and branches
([docs/environment.md](../../docs/environment.md#on-neo4j)). Reload whenever `graph/` changes: a database loaded
before the `news` column has no news, so `sourcing_with_news` decides the news limit with 0.0 ("no supplier in use
has news").

```bash
set -a; . ./.neo4j-local.env; set +a
uv run clm-load-graph examples/tariff/environment.toml
```

## What happened when the models were asked

On 29 October 2026 the question above went to Claude Opus 5.5 (API) and `gemma4:26b` (Ollama, M5 Max). Runs 1–5
used the built-in graph and a server unable to judge news; run 6 Neo4j and a shorter question; runs 7–8 a server
able to judge news, without the models being told. In the Calls column, D is `describe`, S `scenario_sourcing`
and N `scenario_sourcing_with_news`.

| Run | Calls | Time | Recommendation | What went wrong |
|---|---|---|---|---|
| 1 Claude | D, S | not recorded | half to Region B, right | nothing |
| 2 Claude | D, S | 17.8 s | half, right | gave the gain as **$42,400**, its own arithmetic on two tool figures: flagged by the number check; correct, but traced to no step |
| 3 gemma4 | D, S | 20.8 s | half, right | did not say why Kestrel Polymers carries all the supply, though the instructions ask for the dependency behind a rejection |
| 4 gemma4 | D, S | 12.1 s | half, right | same |
| 5 gemma4 | D, S | 19.0 s | half, right | same |
| 6 gemma4, Neo4j | D, S | 14.6 s | half, right | headed the 0% branch "Rejected" though it passed every limit and only ranked lower; the number check checks figures, not words |
| 7 Claude, news | D, N, S | 30.0 s | stay in Region A, right | nothing |
| 8 gemma4, news | D, S | 15.2 s | half, wrong | never ran the news scenario, so missed the strike; every figure right for a question without the news |

- Runs 1–6 made the same two calls and reached the same decision: the scenario decided, the models reported.
- Claude's answers were fuller (tariff rates, the key clients by name and tier, every branch's margin rate and
  tariff cost, the Kestrel dependency) and named what the analysis leaves out: shares between 50% and 60%,
  transition time, switching costs, supplier capacity. gemma4's were shorter (margin, extension, the defect rate once
  in three, the rejection naming Kestrel), did no arithmetic of their own, printed figures as given ("704,900.0")
  and never said what was not modelled. gemma4 was free and local; Claude was about as fast, for an API call.
- With the news, Claude explained both rejections (the strike at 0.92; Kestrel and the strike), ran `sourcing` to
  show half would win without the news, warned unprompted that the award scored 0.79 and the line 0.77 so a person
  should check the strike, and said it cannot test a mix such as motors staying in Region A. gemma4 never looked:
  the framework gives the evidence; looking is the agent's choice. To have the news always checked, offer one
  scenario with the news limit, not two.

Eight runs on a small, made-up graph show the pattern works with both models, not how often an agent gets it right.
A fair next measure would be many varied questions, each scored for the right branch, for naming the rejected
branch and its dependency, and for flagged figures.
