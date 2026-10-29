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
```

## What happened when Claude was asked

The question above was put once to Claude Opus 5.5 through `decision_agent.py` on 29 October 2026.

- **It made two calls**: `describe`, then `scenario_sourcing`.
- **Its recommendation**: move half to Region B, not all of it. It gave the margin, delivery extension and defect
  rate of each branch, and said every key client's contract allows the 2-day extension.
- **Why not everything**: it said moving everything was rejected by the supply-concentration rule. It explained
  that both Region B suppliers depend on Kestrel Polymers, while each Region A supplier relies on a different
  provider.
- **What it couldn't answer**: it said the model tests only 0, 50% and 100%, so it couldn't say whether a share
  between 50% and 60% would do better. It also said switching costs and transition time are not modelled.
- **The number check**: no figure in the answer was one the tools had not returned.

One run on a small, made-up graph shows the pattern works; it is not a measure of how often an agent gets it right.
