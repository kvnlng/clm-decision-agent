# A new site in an expensive area

A made-up bakery chain, Cedar Loaf Bakeries, asks: **"Are we in a position to open a new site in an expensive
area?"** Its finance team writes the rules; the answer goes into a proposal a financial officer verifies, risk
included.

The tariff example shows an agent choosing under hard limits. This one shows the rest of what an officer needs.
The company, sites and neighbourhoods are in `graph/nodes.csv`; the five rules are `rules/*.dmn`, described in
`environment.toml`.

| What | How |
|---|---|
| Rules an agent can explore | `rules` with topic `open_site` lists the five rules, dependencies first: statement, reason, owner, version, the information each needs, and what the asker must supply (`to_ask`: the neighbourhood). `rule` shows one rule's rows |
| Facts and estimates told apart | Finances, rents and site costs are facts. Household income is an estimate (the American Community Survey's median with its 90% margin of error), given as low, central and high with its source and date |
| Risk reported, not judged | Only the owner's rules reject. A site that fails a rule only at the low estimate is kept with a warning; the **headroom** says how far income can fall before a rule fails |
| Approvals | An investment above 400,000 dollars needs the board; the approval is listed, not treated as a rejection |
| A decision record | `clm-agent` writes it beside the answer: question, what was supplied, every rule with its reason, every fact and estimate with values, range and source, each branch's verdicts, warnings, headroom and approvals, and the agent's explanation, marked as not checked |

## Try it

```bash
npx -y @modelcontextprotocol/inspector@2.13.0 --cli uv run clm-server examples/new_site/environment.toml \
    --method tools/call --tool-name rules --tool-arg topic=open_site
npx -y @modelcontextprotocol/inspector@2.13.0 --cli uv run clm-server examples/new_site/environment.toml \
    --method tools/call --tool-name scenario_new_site --tool-arg neighbourhood=tract_4102
uv run clm-agent examples/new_site/environment.toml "Are we in a position to open a new site in an expensive area?"
uv run clm-agent examples/new_site/environment.toml \
    "Are we in a position to open a new site in an expensive area? We are looking at tract_4102."
```

The first question lacks the neighbourhood, so the agent stops and asks for it, naming the rules that need it. The
second runs the scenario and writes the record to `examples/new_site/runs/`.

## What the rules say

| Tract | Median income | Small site | Large site |
|---|---|---|---|
| tract_4102 (41.02) | 221,211 ± 30,505 | **Kept**, board approval; payback and stress test fail at the low estimate; headroom **2%** (stress test) | Rejected: stress test, loan covenant |
| tract_6400 (64) | 223,462 ± 55,486 | **Kept**, board approval; same two warnings; headroom **9.8%** | Rejected: stress test, loan covenant |
| tract_3900 (39) | 211,406 ± 20,879 | Rejected: payback, stress test | Rejected: payback, stress test, loan covenant |
| tract_11500 (115) | 124,613 ± 20,017 | Rejected: payback, stress test | Rejected: payback, stress test, loan covenant |

Tract 64's estimate is the least certain yet has more headroom, because more households live there. Neither site is
safe at the estimate's low end; the record shows which rule would fail and by how much income must fall. The
reserve policy is the risk appetite taken from the data: two profitable years (2024 and 2025) allow up to half the
reserves for expansion; the small site uses 22.5%. A payback of 99 years means the site never pays back.

## What happened when the model was asked

`gemma4:26b` through Ollama, on a Mac, one run each:

| Question | Calls | Time | Answer | What went wrong |
|---|---|---|---|---|
| Vague | `describe`, `rules` | 11 s | Asked for the neighbourhood, naming the payback and stress-test rules: right | nothing |
| With tract_4102 | `describe`, `rules`, `scenario_new_site` | 20 s | Small site, with the board's approval, the income range and source, both warnings, the 2% headroom and why the large site was rejected; no figure flagged: right | led with "Yes, we are in a position", which its own warnings below should temper; the record shows them either way |

Claude was not tried. Two runs of one model are a small test.

## The data

- `data/acs_income.csv`: median household income (table B19013) for four census tracts in Seattle, King County,
  Washington: U.S. Census Bureau, American Community Survey 2023 5-year estimates, 90% margin of error; taken from
  the Bureau's API on 2026-10-30; U.S. government work, public domain. Tracts are keyed by full GEOID (state 53,
  county 033, tract), since a tract number is unique only within its county; `graph/nodes.csv` does the same.
- Everything else (company, finances, rents, households, site costs) is made up. Tracts are named by census number,
  not neighbourhood, so as not to suggest local knowledge the example does not have.
