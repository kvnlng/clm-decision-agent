# Employee handbook

**Not HR, tax or legal advice.** Meridian Logistics, its employees and its handbook are made up for this example.
Only the 2026 IRS limit on 401(k) contributions is real, from
[IRS Notice 2025-67](https://www.irs.gov/pub/irs-drop/n-25-67.pdf).

This is the framework's example of **private documents**: a company's own handbook that no model has seen. The
[non-compete example](../noncompete/README.md) used public law, which Claude already knows, so it could not show
what a local document collection adds. Here the answers exist only in Meridian's handbook and records, and the same
questions are asked six ways: with no tools; with the rules only; with the rules and the handbook; with the handbook
and no rules; and with rules **derived from the handbook** by a model and checked, by Claude or by the local model.

The framework's principle is that a rule is **supplied** (written by the owner) or, when it is not, **derived** from
the documents and **checked** before a person approves it, and that which kind it is is never hidden
([docs/environment.md](../../docs/environment.md#derived-rules-and-facts)). This example measures what deriving costs.

## The questions

Each asked by one made-up employee, naming the company, their id and the date (1 December 2026):

1. What is the company match on my 401(k), and what is the maximum?
2. Can I carry over my unused leave into next year, and how much?
3. I want to start parental leave today. Am I eligible, and for how long?
4. Will the company reimburse my tuition for a $3,900 job-related course, and up to how much?
5. What is the dress code at the Denver office? (No rule covers it: only the handbook answers it.)

## The data

| Part | What |
|---|---|
| `handbook.jsonl` | 12 policies in 13 documents (the 401(k) policy in two editions), their numbered parts as passages: the 401(k) plan summary as restated on 1 January 2026, and its **2024 edition, superseded** (a different match formula, cap, waiting period and vesting, with `effective_until` 2025-12-31), paid time off, parental leave, tuition, dress code by office, and seven unrelated policies with figures of their own. The figures are deliberately not the typical ones |
| `graph/` | Eight employees (hire date, office, salary, contribution rate, unused leave), and the plan's match cap and IRS limit |
| `rules/` | Five DMN tables, each citing the handbook section it comes from: when the match begins, vesting, the carryover limit, parental leave, tuition |
| `handbook_functions.py` | The employee's record as of a date, the calculations (the match, the days carried and forfeited, leave pay, tuition repaid), and one scenario per question |
| `answer_key.toml` | What a good answer says, worked out by hand from the handbook; a test checks the rules and calculations against it, never the reverse |
| `environment_derived.toml` | The same questions with the five rules and the plan's six terms (IRS limit, match cap, and the match's two tiers and rates) as **slots** to derive; the employees' records stay supplied, and `graph/nodes_without_plan_figures.csv` keeps the plan's id without its figures. The match is a generic two-tier `tiered_match`: its shape is still written by hand, as are the other calculations |
| `environment_documents_only.toml` | The employees' records, the calculations and the handbook, and no rules or plan figures: the agent reads every rule and figure itself |
| `derived/` | Claude's derived layer, committed so the variant runs without an API key, as checked again with the stricter check: all 11 entries approved (below). **Its approvals are scripted** (`measure.py: scripted approval of every checked entry, not a person`): no person approved them |

## Running it

PostgreSQL with pgvector and Ollama with `embeddinggemma`, as for noncompete
([docs/environment.md](../../docs/environment.md) covers documents).

```bash
set -a; . ./.postgres-local.env; set +a
uv run clm-load-documents examples/handbook/environment.toml handbook examples/handbook/handbook.jsonl \
    --source "Meridian handbook (made up)"
uv run clm-agent examples/handbook/environment.toml "I work at Meridian Logistics; I'm employee e_2001, and \
today is 1 December 2026. What is the company match on my 401(k), and what is the maximum?"
```

## The six setups

- **No tools**: `clm-agent --no-tools`, the question alone, with a short prompt: answer the employee's question, and
  say so when you do not know. This stands for an employee asking a general chatbot.
- **Rules only**: `environment_without_documents.toml`. The scenarios give the right figures and name the handbook
  sections the rules cite, but not the sections' text, and there is no tool to search or read the handbook. Each
  rule's own statement and reason restate its policy in a sentence, figures included.
- **Rules and handbook**: `environment.toml`. The rules' citations carry the sections' text, and the agent can
  search and read the whole handbook.
- **Handbook only**: `environment_documents_only.toml`. No rules and no plan figures; the agent searches the
  handbook and passes the figures it reads to the calculations itself. With the calculations it is the strongest
  fair competitor to derived rules.
- **Derived (Claude's layer)**: `environment_derived.toml` with the rules and figures Claude derived, answered by
  Claude and by gemma4.
- **Derived locally**: the same with gemma4's own layer, answered by gemma4: nothing leaves the machine.

## Deriving the rules

```bash
uv run clm-derive run examples/handbook/environment_derived.toml handbook --backend claude --as-of 2026-12-01
uv run clm-derive status examples/handbook/environment_derived.toml decisions carryover_limit approved --by NAME
```

`run` gives the model the handbook documents in force on the date (never the superseded 2024 edition), renders the
table it describes, and checks the entry mechanically: the edition, the quote word for word in the cited passages,
every figure in the quote, and only the slot's names. A failed entry is asked for once more with its failures.
Only an approved entry decides; an unfilled slot makes its scenario's branch fail, saying why.
`measure.py derive BACKEND` does this afresh and then approves every checked entry **by script**, standing in for a
person, so the measurement shows what that approval would let through. `measure.py compare` then holds each layer
against a **second key**: the hand-written `rules/*.dmn` (evaluated over every bound in either table, one either
side, and 0; both booleans; never a negative value, since `employee` never gives negative days or years),
`graph/nodes.csv`, and the match's tiers as `company_match` writes them by hand.

After a review the check was made stricter: a range's upper bound (`[0..4]`) now counts as a figure, a quote must
match whole words and numbers ("up to 1" is no longer found in "up to 10"), and a rule's reason is checked like its
statement. In a statement or reason, ISO dates, section references (§ 4(2), Section 6(1), or bare: 6(3), 401(k)) and
a bare year (2026, but not $2026, 2026% or 2026 days) are not policy figures and are not checked; every other number
is. `measure.py recheck BACKEND` checked both kept layers again with it, asking no model: every entry back to
proposed, checked, and the checked ones approved by script. The table is the result. "First try" is how many passed
the old check at once; under the new check no entry was sent back to the model, so it is no guide to a retry now.

| Layer | Passes the check | Right by the second key | **Checked but wrong** | Failed but right |
|---|---|---|---|---|
| Claude (claude-opus-5-5) | 11 of 11 | 11 of 11 | **0** | 0 |
| gemma4:26b | 9 of 11 | 9 of 11 | **2** | 2 |

Under the first, looser check: Claude 11 of 11 passed (8 at the first try), gemma4 10 of 11 (9), with the same
checked-but-wrong counts, 0 and 2.

- **Claude's layer matched every hand-written rule and figure, and passes the stricter check.** Its reasons name
  dates, years and sections ("the 2026 401(k) plan summary, in force on 2026-12-01", "Section 6(1)"), which are not
  policy figures and are exempt; before those exemptions, four of its right tables failed. Under the looser check
  all 11 passed too, three on the second try.
- **gemma4 confused the match's tiers with its rates, and the check passed them.** It gave the first tier as 100%
  of pay and the second as 50% (the key: 3% and 3%), quoting "Meridian matches 100% of the first 3% of pay you
  contribute, and 50% of the next 3%". Every figure is in the quote, so the mechanical check cannot tell a rate from
  a width: only a person reading the entry would. That is exactly what the scripted approval let through, and it
  made one answer wrong (below).
- **gemma4's carryover table now fails** on the stricter check: a row `[0..4]` whose 4 is not in the quote. It gives
  the right answers for every count of years. Its tuition table was right but failed both checks: it gave the
  passages' whole text as their labels.
- The first gemma4 derivation ran with a prompt that showed each document's id inside its heading; gemma4 cited two
  documents by the whole heading and both failed. The prompt now gives `Document id:` on its own line and the layer
  above is the second run. The first run's comparison, kept in `runs/measure/first-gemma4-run-second_key.json` (not
  committed), shows the same two tier figures wrong. gemma4 is also asked without thinking: with thinking, under the
  JSON schema, it generated over 100,000 tokens on the first slot without answering.
- Deriving the 11 slots took under a minute with gemma4 and about two minutes with Claude.

## What was measured

`measure.py run` asked each question once in each setup, on Claude (claude-opus-5-5) and on gemma4:26b through
Ollama, on a Mac: 55 answers, all on the same prompt (the agent's instructions gained a line about derived rules,
so the first three setups were run again). `measure.py score` compared them with the key. Claude read, for each
figure the key lists, the value each answer gives and whether it is stated as fact or hedged; it is told what each
figure is about, never its value. One run each, so a difference of one is noise.

The derived rows were answered from the layers as approved under the first, looser check. Claude's layer is approved
the same under the current check. gemma4's now leaves carryover and tuition unapproved, so it would answer questions
1 and 3 from derived rules (question 1 still with the wrong tiers) and fail 2 and 4 visibly; those answers were not
asked again.

| Setup | Model | Key figures right | Wrong as fact | Not given | Hedged | Verdicts right | Section cited | Seconds (mean) |
|---|---|---|---|---|---|---|---|---|
| No tools | Claude | 1 of 17 | 0 | 16 | 6 | 0 of 4 | 0 of 5 | 8 |
| No tools | gemma4 | 0 of 17 | 0 | 17 | 0 | 0 of 4 | 0 of 5 | 3 |
| Rules only | Claude | 16 of 17 | 0 | 1 | 0 | 4 of 4 | 4 of 5 | 13 |
| Rules only | gemma4 | 16 of 17 | 0 | 1 | 0 | 3 of 4 | 3 of 5 | 13 |
| Rules and handbook | Claude | 17 of 17 | 0 | 0 | 0 | 4 of 4 | 5 of 5 | 14 |
| Rules and handbook | gemma4 | 14 of 17 | 0 | 3 | 0 | 3 of 4 | 3 of 5 | 14 |
| Handbook only | Claude | 16 of 17 | 0 | 1 | 0 | 4 of 4 | 5 of 5 | 17 |
| Handbook only | gemma4 | 10 of 17 | 1 | 6 | 0 | 3 of 4 | 5 of 5 | 24 |
| Derived (Claude's layer) | Claude | 17 of 17 | 0 | 0 | 0 | 4 of 4 | 5 of 5 | 17 |
| Derived (Claude's layer) | gemma4 | 14 of 17 | 0 | 3 | 0 | 4 of 4 | 2 of 5 | 13 |
| Derived locally (gemma4's layer) | gemma4 | 13 of 17 | 2 | 2 | 0 | 4 of 4 | 3 of 5 | 17 |

What it says:

- **Derived rules did as well as hand-written ones here, when Claude derived them** and the looser check passed
  them: 17 of 17 on Claude, the same as rules and handbook. gemma4 answering from Claude's layer gave 14 of 17, as it did with the hand-written rules and
  the handbook, leaving figures out (the match cap and IRS limit; the boot allowance) rather than getting them wrong.
- **Fully local, derivation lost.** gemma4 answering from its own layer stated two wrong figures as fact, on question
  1: a match of $3,680 (the key: $3,220) and a maximum of $6,300 (the key: $4,140), both from the tier figures the
  check passed and the script approved. With supplied or Claude-derived rules no answer stated a key figure wrongly.
- **Handbook only was close on Claude, not on gemma4.** Claude, reading the handbook and calling the calculations
  itself, gave 16 of 17 (it left out the IRS limit); gemma4 gave 10 of 17, one of them wrong as fact:
  it passed the match's rates to `tiered_match` wrongly and answered a match of $32.20. It was also the slowest
  setup. The rules earn their keep most with the local model.
- **The superseded 2024 edition was tested only where nothing filtered it.** Derivation is given the documents in
  force, so it never saw the 2024 edition. In handbook only, Claude read it, said it was replaced on 1 January 2026
  and did not use it. With rules, the agents went straight to the scenarios on questions 1 to 4.
- **Without the company's data there is no answer at all.** No tools: 1 of 17 figures (Claude: the public IRS
  limit), no verdicts; both models said they could not know Meridian's plan, and Claude's six hedged figures were
  what is typical elsewhere. The no-tools prompt tells the model to say when it does not know; a chatbot without that
  instruction was not tested.
- **The handbook adds the one question no rule covers.** With rules only, both models missed question 5's $150 boot
  allowance, which only the handbook gives; with the handbook Claude found it and gemma4 did not this time.
  gemma4 also left out the IRS limit and vested percentage on question 1 with the handbook, giving 14 of 17 against
  16 with rules only: one run, so this may be noise.
- **gemma4 was about as fast as Claude** (13 to 17 s a question) except with the handbook only (24 s). It cited
  handbook sections less often from Claude's derived layer (2 of 5).
- Seconds for Claude's first four setups were measured while Ollama was busy with another job, so they may be a
  little high; the differences are within a run's noise anyway.
- **Figures no tool returned** were mostly the models' own arithmetic and the handbook's text quoted ("$920 a year
  in match", "$3,600 more in total"); they are flagged in each answer file and not graded, since the key does not
  hold them.
- What is still written by hand: the employees' records, the two-tier shape of the match, the other calculations
  and the scenarios. Only the five tables and six plan figures were derived. The scripted approval is not a person.

The reader of the figures is Claude, choosing from what the answer says; it may read an answer kindly. The key and
the policies were written by the same author, which makes the key easy to check against the handbook but says
nothing about a real company's handbook, which is longer, contradicts itself and changes more often.
