# Employee handbook

**Not HR, tax or legal advice.** Meridian Logistics, its employees and its handbook are made up. Only the 2026 IRS
limit on 401(k) contributions is real, from [IRS Notice 2025-67](https://www.irs.gov/pub/irs-drop/n-25-67.pdf).

This is the framework's example of **private documents**: a handbook no model has seen, where the
[non-compete example](../noncompete/README.md) used public law Claude already knows. A rule is **supplied** by the
owner or **derived** from the documents by a model and **checked** before a person approves it, and it is never
hidden which ([docs/environment.md](../../docs/environment.md#derived-rules-and-facts)). This example measures what
deriving costs. The measured results, with their conditions and tables, are in [measurements.md](measurements.md).

## The questions

Each is asked by one made-up employee, who names the company, their id and the date (1 December 2026):

1. What is the company match on my 401(k), and what is the maximum?
2. Can I carry over my unused leave into next year, and how much?
3. I want to start parental leave today. Am I eligible, and for how long?
4. Will the company reimburse my tuition for a $3,900 job-related course, and up to how much?
5. What is the dress code at the Denver office? (No rule covers it: only the handbook answers it.)

## The data

| Part | What |
|---|---|
| `handbook.jsonl` | 12 policies in 13 documents, their numbered parts as passages: the 401(k) plan summary as restated on 1 January 2026 and its **superseded 2024 edition** (a different match formula, cap, waiting period and vesting; `effective_until` 2025-12-31), paid time off, parental leave, tuition, dress code by office, and seven unrelated policies with figures of their own. The figures are deliberately not the typical ones |
| `graph/` | Eight employees (hire date, office, salary, contribution rate, unused leave) and the plan's match cap and IRS limit. `nodes_without_plan_figures.csv` keeps the plan's id without its figures, for deriving |
| `rules/` | Five DMN tables: when the match begins, vesting, the carryover limit, parental leave, tuition. Each rule in `environment.toml` cites the handbook document it comes from |
| `handbook_functions.py` | The employee's record as of a date, the calculations (match, days carried and forfeited, leave pay, tuition repaid) and one scenario per question |
| `answer_key.toml` | What a good answer says, worked out by hand from the handbook; a test checks the rules and calculations against it, never the reverse |
| `derived/` | Claude's derived layer, committed so the derived setup runs without an API key: all 11 entries approved under the current check (below). **The approvals are scripted, not a person's** (`measure.py: scripted approval of every checked entry, not a person`) |

## The setups

| Setup | Environment | What the agent has |
|---|---|---|
| No tools | `clm-agent --no-tools` | The question and a short prompt: answer, and say so when you do not know. It stands for a general chatbot |
| Rules only | `environment_without_documents.toml` | The scenarios give the right figures and name the handbook sections the rules cite, not their text; no tool searches or reads the handbook. Each rule's statement and reason restate its policy, figures included |
| Rules and handbook | `environment.toml` | The rules' citations carry the sections' text; the agent can search and read the whole handbook |
| Handbook only | `environment_documents_only.toml` | The employees' records, the calculations and the handbook; no rules or plan figures. The agent reads every rule and figure and passes them to the calculations itself. The strongest fair competitor to derived rules |
| Derived (Claude's layer) | `environment_derived.toml` | The five rules and the plan's six terms (IRS limit, match cap, the match's two tiers and rates) as **slots**, filled by Claude; answered by Claude and by gemma4. The employees' records stay supplied; the match is a generic two-tier `tiered_match` whose shape, like the other calculations and the scenarios, is written by hand |
| Derived locally | `environment_derived.toml` | The same with gemma4's own layer, answered by gemma4: nothing leaves the machine |
| Fewer tools | a copy of `environment.toml` in `runs/measure/fewer_tools/` | Rules and handbook behind `[serve] tools = ["describe", "scenario_*", "search_*", "document_*"]`, written by `measure.py` ([Fewer tools](measurements.md#fewer-tools)) |

## Running it

Needs PostgreSQL with pgvector and Ollama with `embeddinggemma` ([docs/environment.md](../../docs/environment.md)).

```bash
set -a; . ./.postgres-local.env; set +a
```

```bash
uv run clm-load-documents examples/handbook/environment.toml handbook examples/handbook/handbook.jsonl \
    --source "Meridian handbook (made up)"
```

```bash
uv run clm-agent examples/handbook/environment.toml "I work at Meridian Logistics; I'm employee e_2001, and \
today is 1 December 2026. What is the company match on my 401(k), and what is the maximum?"
```

## Deriving the rules

```bash
uv run clm-derive run examples/handbook/environment_derived.toml handbook --backend claude --as-of 2026-12-01
```

```bash
uv run clm-derive status examples/handbook/environment_derived.toml decisions carryover_limit approved --by NAME
```

`run` gives the model the documents in force on the date (never the 2024 edition), renders the table it describes
and checks four things: the edition; that the quote is whole and word for word in the cited passages ("up to 1" is
not in "up to 10"); that every figure in the statement, reason and rows is in the quote, a range's upper bound
included, except ISO dates, section references (§ 4(2), 6(3), 401(k)) and bare years (2026, not $2026 or 2026 days);
and that only the slot's names are used. A failed entry is asked once more with its failures. Only an approved
entry decides; an unfilled slot fails its scenario's branch, saying why.

Claude's layer passed the check 11 of 11 and was right by the second key 11 of 11; gemma4:26b's passed 9 of 11, and
its two wrong tiers (100% and 50% of pay, where the key says 3% and 3%) passed the check, because **the check cannot
tell a rate from a width**. The derivation table, the `measure.py` subcommands and the first gemma4 run:
[measurements.md](measurements.md#deriving-the-rules).

## A second reading

The check proves each figure is in the quote, not that it is in the right place. A **second reading** by another
model, given the cited document and the slot but never the first answer, fails an entry the two readings disagree on
or that it cannot produce ([docs/environment.md](../../docs/environment.md#a-second-reading)).
`environment_derived.toml` names no second reader, so the committed layer is checked as before. Claude, granite4:micro
and qwen3:14b each read gemma4's layer again, and gemma4 read Claude's. Claude caught both wrong tiers and flagged no
right entry; the local readers flagged right entries, and **qwen3:14b agreed with gemma4's wrong second tier, so the
entry passed both checks: agreement is not proof**. The table and each reader's detail:
[measurements.md](measurements.md#a-second-reading).

## What was measured

Each question was asked once in each setup, on Claude (claude-opus-5-5) and on gemma4:26b through Ollama, on a Mac;
the tables and the conditions are in [measurements.md](measurements.md#what-was-measured). Claude-derived rules did
as well as hand-written ones; gemma4 from Claude's layer left figures out (the match cap, IRS limit and boot
allowance) rather than getting them wrong. **Fully local, derivation lost**: gemma4 from its own layer stated two
wrong figures as fact on question 1, from the tiers the check passed. With supplied or Claude-derived rules no answer
stated a key figure wrongly. **Handbook only was close on Claude, not on gemma4**, which passed the rates to
`tiered_match` wrongly and was slowest: the rules earn their keep most with the local model. Only the handbook gives
question 5's boot allowance: Claude found it; gemma4 did not this time. A shorter tool list
([Fewer tools](measurements.md#fewer-tools)) neither helped nor hurt the local model.
