# Non-competes

**Not legal advice.** This example is a test of the framework, written by software engineers, not lawyers. No lawyer
has reviewed its rules, its answer key or its answers. Every answer is a starting point that cites its sources so
that someone qualified can check them; none is a ruling on any real agreement.

Meridian Logistics, a made-up company, employs people in eleven states under one standard non-compete agreement,
`standard_v3`: 24 months, nationwide, no garden leave, Texas law named, signed 1 March 2024. Its general counsel
asks three questions:

1. **Does standard_v3 hold in each state, and what must change?** A verdict per state, from its statutes.
2. **In Texas, is a 2-year, nationwide non-compete for a sales manager reasonable,** for one who serves Texas,
   Oklahoma and Louisiana, and how should it be narrowed? Texas reforms a covenant broader than needed, so this
   is answered from how Texas courts treated similar covenants.
3. **Which law governs a Colorado employee whose contract names Texas law?** Colorado's choice-of-law rule.

It is the framework's example of **local documents**: statutes and about 4,800 court opinions in PostgreSQL with
pgvector, searched by meaning and by words, read by passage, and followed by citation, with the framework's own
tools. The example writes no tool code. Rules decide; retrieval only finds the evidence the rules and the answer
cite.

## The data

| Part | What | Source | Serves |
|---|---|---|---|
| `statutes.jsonl` | 39 sections of the non-compete statutes of TX, CA, CO, WA, IL, MA, MN, FL, OK and ND, one document per section, its subsections as passages. Washington's appear twice: in force until 29 June 2027, and from 30 June 2027. New York has none | Each state's official publisher (for example Texas Legislature Online, California Legislative Information, Online Sunshine for Florida, Washington's RCW site, OSCN for Oklahoma), collected by hand | 1, 3 |
| `data/cases.jsonl` (built, not committed) | 4,807 published opinions: those on non-competes in those states, New York and the Fifth Circuit applying Texas law, and the opinions in those courts that cite them; 113,481 passages and 130,985 citations between them | The [Caselaw Access Project](https://static.case.law) (Harvard Law School Library), free of restrictions, to about 2020 | 2 |
| `graph/` | The made-up company: the agreement, staff and lowest pay per state, employees, and each state's 2026 pay threshold with its published source | Made up; thresholds from the Colorado and Washington labour departments | 1, 3 |
| `rules/` | DMN tables: each state's law with its citation, choice of law, a court's weight, what counts as a similar case, how likely a covenant is enforced | Written for the example, each row citing its statute | 1, 2, 3 |
| `answer_key.toml` | What a good answer says, and cites, for each question | Read from the cited texts; checked mechanically against them (every citation is in the collection, every figure is in its source), not by a lawyer | measuring |
| `labels/` | Passages labelled yes or no for the two judgements below, each quoting the court's own words | Texas opinions for one, Texas and California for the other, chosen by hand | measuring |

The opinions are mostly Texas (1,798) and mostly 1990 to 2019. Florida has 764, California 570, New York 524,
Illinois 507; Oklahoma only 25 and Colorado 68. **Recent laws have no cases:** Minnesota's 2023 ban, Colorado's
2022 and Washington's 2026 changes postdate the collection, so for those states the statutes are the whole answer.

A local model (gemma4:26b) read each non-compete opinion's facts: the employee's role, the length, the area, the
outcome and what a court narrowed it to. A fact is kept only if the passage it names contains the words read, but
a passage can contain the words and still be misread. Checking the labels found several: an injunction's 2 years
read as the covenant's length, post-judgment interest read as a covenant reformed, a contractor read as an
employee. Of the 626 opinions with an outcome, 395 also have a length.

## Running it

PostgreSQL (18 here) with pgvector (Homebrew, no Docker) and Ollama with `embeddinggemma` and `gemma4:26b`.
`.postgres-local.env` holds `CLM_DOCUMENTS_PASSWORD`; [docs/environment.md](../../docs/environment.md) covers
documents.

```bash
uv sync --extra documents --extra agent
set -a; . ./.postgres-local.env; set +a
uv run clm-load-documents examples/noncompete/environment.toml statutes examples/noncompete/statutes.jsonl \
    --source "State legislatures, collected 2026-11-03"
uv run python examples/noncompete/fetch_cases.py fetch    # about 11 GB read, one volume at a time; keeps about 0.7 GB
uv run python examples/noncompete/fetch_cases.py build    # reads each opinion's facts locally: about an hour
uv run clm-load-documents examples/noncompete/environment.toml cases examples/noncompete/data/cases.jsonl \
    --source "Caselaw Access Project, fetched 2026-11-03"

npx -y @modelcontextprotocol/inspector@2.13.0 --cli \
    sh -c 'set -a; . ./.postgres-local.env; exec uv run clm-server examples/noncompete/environment.toml' \
    --method tools/call --tool-name scenario_agreement_by_state --tool-arg agreement=standard_v3
uv run clm-agent examples/noncompete/environment.toml \
    "In Texas, is a 2-year, nationwide non-compete for a sales manager reasonable?"
```

`environment_without_documents.toml` is the same environment with no tools to search, read or follow the
documents; the rules still use them. It exists to measure what the document tools add.

## What was measured

`measure.py run` asked each question once with each environment file, on Claude (claude-opus-5-5) and on
gemma4:26b through Ollama, on a Mac; `measure.py score` compared the answers with `answer_key.toml`. One run each,
so a difference of one is noise. Verdicts were read from the answers by Claude, choosing from a fixed list; key
citations, citations no tool returned, and cases found are counted by the script.

| Question | Documents | Model | Verdicts right | Key citations cited | Citations no tool returned | Cases cited and found | Tool calls | Seconds |
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

What it says:

- **For questions 1 and 3 the document tools made no difference.** The rules carry the statutes and their
  citations, so both models got every state's verdict from `scenario_agreement_by_state` either way. The three key
  citations every answer missed are New York's case (the rules cite none for New York), 820 ILCS 90/20 and
  Minnesota's subdivision 2.
- **For question 2 they made the answer somewhat better sourced, not different.** Every run reached "narrow it".
  With the documents Claude cited 13 cases the tools returned, against 9 without, where only the scenario's own
  case list was available; with them it also cited one case, 818 S.W.2d 381, that no tool returned.
- **Claude was right more often than gemma4, and as fast or faster.** gemma4 matched Claude on questions 1 and 2, but on
  question 3 passed `works: "contract"` to the choice-of-law rule once and read Colorado's threshold as "void"
  once.
- **Question 2 is too big a result for a local model.** `scenario_narrow_to_enforce` returns about 0.7 MB (every
  branch's similar cases, with their steps). Ollama silently cuts a conversation to its context window. In a first
  run, before the review's fixes, the result was about 1 MB and gemma4 lost the instructions and the question and
  answered "please provide the question"; in the second it answered, but took over 7 minutes. Neither the scenario
  nor the agent warns yet when a result cannot fit.

### The two judgements

Question 2 has two yes-or-no judgements made by the local CLM: whether a narrowed covenant is still **broader than
needed**, and whether a citing passage **departs** from the case it cites ("still good law?"). `judge_labels.py`
measured them on the passages in `labels/`, each labelled from a court's own words:

- **broader than needed** (7 yes, 4 no) is worse than chance: it scored the covenants courts enforced higher, on
  average, than those they narrowed, ordering only 6 of the 28 yes-no pairs rightly. Courts weigh facts outside
  the covenant's text; the judgement sees only the text.
- **departs** (8 yes, 5 no) has some signal, the top five all real departures, 28 of 40 pairs ordered rightly,
  but no threshold separates them.

So no rule reads either. `broader_than_needed` is reported in the figures and is not evidence; `departs` only
orders each weighty case's citing passages, the most critical first.

A case counts as **similar** (`rules/similar_case.dmn`) when its facts are known, its court is binding or strong
(a federal court is only persuasive), and its role is in the same line of work: the asked role's first word,
"sales" of "sales manager", starts a word in the role the opinion names. Words like "senior" and "general" are
skipped, so "senior sales engineer" looks for "sales". That is a crude rule over free text; an "account
executive" who sells is missed. A case's area counts as nationwide when it names the country as a whole, not a
radius or a region of it ("Southeastern United States"). The label sets are small, eleven and thirteen passages.
