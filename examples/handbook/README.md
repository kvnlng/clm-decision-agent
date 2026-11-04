# Employee handbook

**Not HR, tax or legal advice.** Meridian Logistics, its employees and its handbook are made up for this example.
Only the 2026 IRS limit on 401(k) contributions is real, from
[IRS Notice 2025-67](https://www.irs.gov/pub/irs-drop/n-25-67.pdf).

This is the framework's example of **private documents**: a company's own handbook that no model has seen. The
[non-compete example](../noncompete/README.md) used public law, which Claude already knows, so it could not show
what a local document collection adds. Here the answers exist only in Meridian's handbook and records, and the same
questions are asked three ways: with no tools, with the rules only, and with the rules and the handbook.

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

## The three setups

- **No tools**: `clm-agent --no-tools`, the question alone, with a short prompt: answer the employee's question, and
  say so when you do not know. This stands for an employee asking a general chatbot.
- **Rules only**: `environment_without_documents.toml`. The scenarios give the right figures and name the handbook
  sections the rules cite, but not the sections' text, and there is no tool to search or read the handbook. Each
  rule's own statement and reason restate its policy in a sentence, figures included.
- **Rules and handbook**: `environment.toml`. The rules' citations carry the sections' text, and the agent can
  search and read the whole handbook.

## What was measured

`measure.py run` asked each question once in each setup, on Claude (claude-opus-5-5) and on gemma4:26b through
Ollama, on a Mac: 30 answers. `measure.py score` compared them with the key. Claude read, for each figure the key
lists, the value each answer gives and whether it is stated as fact or hedged; it is told what each figure is about,
never its value. One run each, so a difference of one is noise.

| Setup | Model | Key figures right | Wrong as fact | Not given | Hedged | Verdicts right | Section cited | Seconds (mean) |
|---|---|---|---|---|---|---|---|---|
| No tools | Claude | 1 of 17 | 0 | 16 | 6 | 0 of 4 | 0 of 5 | 8 |
| No tools | gemma4 | 0 of 17 | 0 | 17 | 0 | 0 of 4 | 0 of 5 | 7 |
| Rules only | Claude | 16 of 17 | 0 | 1 | 0 | 4 of 4 | 4 of 5 | 14 |
| Rules only | gemma4 | 16 of 17 | 0 | 1 | 0 | 3 of 4 | 3 of 5 | 14 |
| Rules and handbook | Claude | 17 of 17 | 0 | 0 | 0 | 4 of 4 | 5 of 5 | 15 |
| Rules and handbook | gemma4 | 15 of 17 | 0 | 2 | 0 | 4 of 4 | 1 of 5 | 14 |

What it says:

- **No answer stated a wrong figure as fact, in any setup.** This counts the key's 17 figures. Further figures an
  answer states as fact are not graded by the script, since the key does not hold them; they are counted as returned
  by a tool or not. The four that no tool returned (the 90th day, twice; $920; $3,600) were checked by hand and are
  right. With no tools both models said they could not know Meridian's plan; Claude gave only the public IRS limit
  ($24,500, right), and its six hedged figures were what is typical elsewhere ("often 30 days").
  The failure this example was built to catch did not happen, but the no-tools prompt tells the model to say when it
  does not know, which may be why; a chatbot without that instruction was not tested.
- **Without the company's data there is no answer at all.** No tools: 1 of 17 figures, no verdicts. That gap, not
  wrong figures, is what the local data closes.
- **The rules carry the figures; the handbook adds the one question no rule covers.** On questions 1 to 4 Claude
  gave the same figures with rules only and with the handbook (16 of 16 each); gemma4 gave 16 with rules only and 14
  with the handbook, leaving out the IRS limit and the days forfeited. The difference is question 5, the Denver dress code, which
  only the handbook answers (both models found the $150 boot allowance), and the policy text the answers can quote.
- **gemma4 was as fast as Claude here**, unlike the non-compete example, because these results are small. It cited
  handbook sections less often (1 of 5 with the handbook, where it named tools and steps instead) and left two
  figures out.
- **The superseded 2024 edition was never tested.** On questions 1 to 4 every agent went straight to the scenario
  and never searched the handbook (Claude read the leave policy once); only question 5 searched. No answer quoted the
  old edition, but none had the chance to: a 401(k) question no rule covers would be needed to test it.
- **Figures no tool returned** were mostly the models' own arithmetic ("$920 a year more at 6%", "$3,600 more for
  15 weeks"), which the agent's instructions forbid; each is flagged in its answer file.

The reader of the figures is Claude, choosing from what the answer says; it may read an answer kindly. The key and
the policies were written by the same author, which makes the key easy to check against the handbook but says
nothing about a real company's handbook, which is longer, contradicts itself and changes more often.
