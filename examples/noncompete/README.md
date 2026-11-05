# Non-competes

**Not legal advice.** Software engineers wrote this test of the framework; no lawyer has reviewed its rules, answer
key or answers. Each cites its sources for someone qualified to check.

Meridian Logistics, a made-up company, employs people in eleven states under one non-compete, `standard_v3`: 24
months, nationwide, no garden leave, Texas law named, signed 1 March 2024. Its general counsel asks:

1. **Does standard_v3 hold in each state, and what must change?** From the statutes.
2. **In Texas, is a 2-year, nationwide non-compete reasonable for a sales manager** serving Texas, Oklahoma and
   Louisiana, and how should it be narrowed? From similar cases, since Texas reforms an overbroad covenant.
3. **Which law governs a Colorado employee whose contract names Texas law?** Colorado's choice-of-law rule.

The framework's **local documents** example: about 4,800 opinions and the statutes in PostgreSQL with pgvector,
read with built-in tools, no tool code written. Rules decide; retrieval only finds the evidence they cite.

## The data

| Part | What | Source | Serves |
|---|---|---|---|
| `statutes.jsonl` | 39 sections of TX, CA, CO, WA, IL, MA, MN, FL, OK and ND non-compete statutes, subsections as passages; Washington's twice (until and from 30 June 2027); none for New York | Official state publishers | 1, 3 |
| `data/cases.jsonl` (built, not committed) | 4,807 published non-compete opinions of those states, New York and the Fifth Circuit (Texas law), plus opinions citing them: 113,481 passages, 130,985 citations; mostly Texas (1,798), then Florida 764, California 570, New York 524, Illinois 507, Colorado 68, Oklahoma 25; mostly 1990 to 2019 | [Caselaw Access Project](https://static.case.law) (Harvard Law School Library), free of restrictions, to about 2020 | 2 |
| `graph/` | The company: agreement, staff and lowest pay per state, employees, 2026 pay thresholds | Made up; thresholds from state labour departments | 1, 3 |
| `rules/` | DMN tables: each state's law, choice of law, court weight, similar case, likely enforced | Written here, each row citing its statute | 1, 2, 3 |
| `answer_key.toml` | What a good answer says and cites | The cited texts; checked mechanically, not by a lawyer | measuring |
| `labels/` | Passages labelled yes or no for [the two judgements](#the-two-judgements) | Texas and California opinions, by hand | measuring |

**Recent laws have no cases:** Minnesota's 2023 ban and Colorado's 2022 and Washington's 2026 changes postdate
the collection, so there the statutes are the whole answer.

gemma4:26b read each non-compete opinion's facts; a fact is kept only if the passage it names contains the words
read, and misreadings still happen. 626 opinions have an outcome, 395
of them a length. `build` appends to `data/facts-MODEL.jsonl` and skips opinions already there, even ones read
as no facts; delete a line, or the file, to read again.

## Running it

PostgreSQL 18 with pgvector (Homebrew, no Docker) and Ollama with `embeddinggemma` and `gemma4:26b`.
`.postgres-local.env` (git-ignored) holds `CLM_DOCUMENTS_PASSWORD`, named in `environment.toml`
([docs/environment.md](../../docs/environment.md)).

```bash
uv sync --extra documents --extra agent
set -a; . ./.postgres-local.env; set +a
uv run clm-load-documents examples/noncompete/environment.toml statutes examples/noncompete/statutes.jsonl \
    --source "State legislatures, collected 2026-11-03"
uv run python examples/noncompete/fetch_cases.py fetch    # reads about 11 GB, a volume at a time; keeps about 0.7 GB
uv run python examples/noncompete/fetch_cases.py build    # reads each opinion's facts locally: about an hour
uv run clm-load-documents examples/noncompete/environment.toml cases examples/noncompete/data/cases.jsonl \
    --source "Caselaw Access Project, fetched 2026-11-03"

npx -y @modelcontextprotocol/inspector@2.13.0 --cli \
    sh -c 'set -a; . ./.postgres-local.env; exec uv run clm-server examples/noncompete/environment.toml' \
    --method tools/call --tool-name scenario_agreement_by_state --tool-arg agreement=standard_v3
uv run clm-agent examples/noncompete/environment.toml \
    "In Texas, is a 2-year, nationwide non-compete for a sales manager reasonable?"
```

`environment_without_documents.toml` is the same without the tools to search, read or follow the documents; the
rules still use them.

## What was measured

`measure.py run` asked each question once per environment file, on Claude (claude-opus-5-5) and gemma4:26b through
Ollama, on a Mac; `measure.py score` compared answers with `answer_key.toml`, Claude reading the verdicts from a
fixed list. One run each: a difference of one is noise.

| Question | Documents | Model | Verdicts right | Key citations | Unreturned citations | Cases found | Calls | Seconds |
|---|---|---|---|---|---|---|---|---|
| 1 | with | Claude | 11/11 | 14/17 | 0 | | 5 | 30 |
| 1 | without | Claude | 11/11 | 14/17 | 0 | | 3 | 27 |
| 1 | with | gemma4 | 11/11 | 14/17 | 1 | | 2 | 38 |
| 1 | without | gemma4 | 11/11 | 14/17 | 0 | | 2 | 20 |
| 2 | with | Claude | 1/1 | 2/5 | 1 | 13 | 5 | 54 |
| 2 | without | Claude | 1/1 | 1/5 | 0 | 9 | 5 | 48 |
| 2 | with | gemma4 | 1/1 | 0/5 | 0 | 3 | 5 | 439 |
| 2 | without | gemma4 | 1/1 | 1/5 | 0 | 0 | 4 | 429 |
| 3 | with | Claude | 2/2 | 3/3 | 0 | | 6 | 26 |
| 3 | without | Claude | 2/2 | 3/3 | 0 | | 5 | 19 |
| 3 | with | gemma4 | 1/2 | 3/3 | 0 | | 8 | 23 |
| 3 | without | gemma4 | 0/2 | 0/3 | 0 | | 6 | 30 |

Questions 1 and 3 come from the rules, so the document tools changed nothing; every answer missed New York's
case (the rules cite none), 820 ILCS 90/20 and Minnesota's subdivision 2. gemma4's question 3 errors: `works: "contract"` passed to the choice-of-law rule; Colorado's threshold read
as "void". **Question 2's result is too big for a local model:** `scenario_narrow_to_enforce` returns about 0.7 MB
and Ollama silently cuts a conversation to its context window; a first run (about 1 MB, before the review's fixes)
lost the question. Nothing warns yet when a result cannot fit.

### The two judgements

Question 2's two yes-or-no judgements by the local CLM, measured by `judge_labels.py` on `labels/`:

| Judgement | Labels | Pairs ordered rightly | Result |
|---|---|---|---|
| `broader_than_needed` (a narrowed covenant still overbroad?) | 7 yes, 4 no | 6 of 28 | Worse than chance: courts weigh facts outside the text. Reported, not evidence |
| `departs` (a citing passage departs from the case cited?) | 8 yes, 5 no | 28 of 40 | Some signal, no separating threshold. Only orders citing passages, most critical first |

No rule reads either. A case is **similar** (`rules/similar_case.dmn`) when its facts are known, its court is
binding or strong (federal courts only persuasive) and its role is in the same line of work: the asked role's
first word ("sales") starts a word in the opinion's role, skipping words like "senior": crude, an "account
executive" who sells is missed. An area is nationwide when it names the whole country, not a radius or region.
