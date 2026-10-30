# A new site in an expensive area

A made-up bakery chain, Cedar Loaf Bakeries, asks: **"Are we in a position to open a new site in an expensive
area?"** Its finance team writes the rules; the answer goes into a proposal that a financial officer verifies,
including the risk behind it.

The tariff example shows an agent choosing among alternatives under hard limits. This one shows the rest of what an
officer needs:

- **Rules an agent can explore.** `rules` with the topic `open_site` lists the five rules that govern opening a site,
  dependencies first, with what each says, why it exists, its owner and version, the information each needs, and
  what the person asking must supply (`to_ask`: the neighbourhood). `rule` shows one rule with its table's rows.
- **Facts and estimates told apart.** The company's finances, rents and site costs are facts. Household income is an
  estimate: the American Community Survey's median with its 90% margin of error, so each figure arrives with a low,
  central and high, its source and its date.
- **Risk reported, not judged.** Only the owner's rules reject. A site that passes at the central estimate but fails a
  rule at the low one is kept with a warning, and the **headroom** says how far income can fall before a rule fails.
- **Approvals.** An investment above 400,000 dollars needs the board: listed, not a rejection.
- **A decision record.** `clm-agent` writes it beside the answer: the question, what was supplied, every rule with
  its reason, every fact with its values and every estimate with its range and source, each branch's verdicts,
  warnings, headroom and approvals, and the agent's own explanation, marked as not checked.

## Try it

```bash
uv run clm-call examples/new_site/environment.toml rules '{"topic": "open_site"}'
uv run clm-call examples/new_site/environment.toml scenario_new_site '{"neighbourhood": "tract_4102"}'
uv run clm-agent examples/new_site/environment.toml "Are we in a position to open a new site in an expensive area?"
uv run clm-agent examples/new_site/environment.toml \
    "Are we in a position to open a new site in an expensive area? We are looking at tract_4102."
```

The first question lacks the neighbourhood, so the agent stops and asks for it, naming the rules that need it. The
second runs the scenario; the record is written to `examples/new_site/runs/`, beside the answer.

## What the rules say

| Neighbourhood (census tract) | Median household income | Small site | Large site |
|---|---|---|---|
| tract_4102 (Tract 41.02) | 221,211 ± 30,505 | **Kept**; the board must approve; at the low estimate payback and the stress test fail; income can fall **2%** before the stress test fails | Rejected: stress test, loan covenant |
| tract_6400 (Tract 64) | 223,462 ± 55,486 | **Kept**; the board must approve; the same two warnings at the low estimate; income can fall **9.8%** before the stress test fails | Rejected: stress test, loan covenant |
| tract_3900 (Tract 39) | 211,406 ± 20,879 | Rejected: payback, stress test | Rejected: payback, stress test, loan covenant |
| tract_11500 (Tract 115) | 124,613 ± 20,017 | Rejected: payback, stress test | Rejected: payback, stress test, loan covenant |

Tract 64's estimate is the least certain, yet it has more headroom than Tract 41.02, because more households live
there. Neither site is safe at the census estimate's low end: an officer reading the record sees exactly which rule
would fail, and by how much income would have to fall.

The reserve policy is the risk appetite taken from the data: two profitable years (2024 and 2025) allow up to half
the reserves for expansion, and the small site uses 22.5%. A payback of 99 years means the site never pays back.

## What happened when the model was asked

gemma4:26b through Ollama, on a Mac, one run each:

- **The vague question:** it called `describe` and `rules`, then stopped and asked for the neighbourhood, naming the
  payback and stress-test rules as the ones that need it (2 calls, 11 s).
- **With tract_4102:** `describe`, `rules`, `scenario_new_site` (3 calls, 20 s). It recommended the small site,
  named the board's approval, gave the income estimate's range and source, both warnings and the 2% headroom, and
  why the large site was rejected. The number check flagged no figure. It led with "Yes, we are in a position",
  which a careful reader would temper with the warnings it listed below; the record shows them either way.

Claude was not tried on this example. Two runs of one model are a small test.

## The data

- `data/acs_income.csv`: median household income (table B19013) for four census tracts in Seattle, King County,
  Washington, from the U.S. Census Bureau's American Community Survey 2023 5-year estimates, with the 90% margin of
  error. Taken from the Census Bureau's API on 2026-10-30. U.S. government work, in the public domain.
- Everything else (the company, its finances, rents, households and site costs) is made up.
- The tracts are named by their census numbers, not by neighbourhood names, so as not to suggest local knowledge the
  example does not have.
