# Employee handbook: measurements

This page holds the measured results for [the handbook example](README.md): the derivation runs, the second
readings and the answers in each setup, with their conditions.

## Deriving the rules

`measure.py derive BACKEND` approves every checked entry **by script**, to show what a person's approval would let
through. `recheck` sets each entry back to proposed and checks it again with the current check, asking no model.
`compare` holds each layer against a **second key**: `rules/*.dmn` over every bound in either table, one either
side and 0, never negative, and both booleans; `graph/nodes.csv`; and the tiers as `company_match` writes them.
gemma4 is asked without thinking (with it, under the JSON schema, it ran past 100,000 tokens on the first slot). Its
layer is a second run. The first run, with document ids inside headings rather than on a `Document id:` line, cited
two documents by the whole heading, so both entries failed, and had the same two tier figures wrong
(`runs/measure/first-gemma4-run-second_key.json`, not committed).

| Layer | Passes the check | Right by the second key | **Checked but wrong** | Failed but right | Under the first, looser check | Notes |
|---|---|---|---|---|---|---|
| Claude (claude-opus-5-5) | 11 of 11 | 11 of 11 | **0** | 0 | 11 of 11 passed, 8 at the first try; wrong 0 | Reasons name dates and sections ("in force on 2026-12-01", "Section 6(1)"); four right tables failed before those were exempt |
| gemma4:26b | 9 of 11 | 9 of 11 | **2** | 2 | 10 of 11 passed, 9 at the first try; wrong 2 | Wrong: the tiers as 100% and 50% of pay (key: 3% and 3%). Failed but right: carryover (a row `[0..4]`, 4 not in the quote) and tuition (the passages' whole text as labels; failed both checks) |

No entry was sent back to the model under the current check, so the first-try counts are no guide to a retry now.
Deriving the 11 slots took under a minute with gemma4, about two minutes with Claude. **The check cannot tell a rate
from a width**: gemma4's tiers quote "Meridian matches 100% of the first 3% of pay you contribute, and 50% of the
next 3%", every figure is in it, the script approved them, and one answer was wrong (below).

## A second reading

`measure.py second FIRST BACKEND MODEL` read each kept layer again, leaving it as it was, and counted against the
second key. A "flagged" entry is a right one the second reading disagreed with, a false alarm a person must look at.
One run each; timings not recorded.

| First reader | Second reader | Wrong entries caught | Right entries flagged | Pass both | Pass both but wrong | What the second reader read |
|---|---|---|---|---|---|---|
| gemma4:26b | Claude (claude-opus-5-5) | **2 of 2** | 0 of 9 | 7 of 11 | 0 | Both tiers ("100 against 3", "50 against 3") |
| gemma4:26b | granite4:micro | 2 of 2 | **5 of 9** | 3 of 11 | 0 | Widths right (3 and 3) but both rates as 3 too, so all four tier figures disagreed; three tables with no row for some years or days; other vesting percentages |
| gemma4:26b | qwen3:14b (9.3 GB) | **1 of 2** | 2 of 9 | 6 of 11 | **1** | First tier caught (100 against 3); second tier as 50% of pay, as gemma4 had; a right rate flagged (3 read as the rate); one table does not parse (`>& 4`) |
| Claude (claude-opus-5-5) | gemma4:26b | none to catch | **3 of 11** | 8 of 11 | 0 | Rates and widths confused both ways (a first rate of 3, a second width of 50); `any value` in a tuition cell, which does not parse |

Claude as second reader is not local: the documents go to Claude. granite4:micro's false alarms include 4 entries
the mechanical check passes, so one more local model was tried. **qwen3:14b agreed with gemma4's wrong second tier,
so the entry passed both checks: agreement is not proof**, and a fully local second reading did not close the gap. A
person still reviews every entry; the second reading changes which ones they are told to look at. With Claude
reading gemma4's layer, those are the two wrong tiers and the carryover and tuition tables the check already failed.

## What was measured

`measure.py run` asked each question once in each setup, on Claude (claude-opus-5-5) and on gemma4:26b through
Ollama, on a Mac: 55 answers on one prompt (the first three setups were run again when the agent's instructions
gained a line about derived rules). `measure.py score` had Claude read, for each figure the key lists, the value an
answer gives and whether it is fact or hedged, told what the figure is about but never its value. One run each, so a
difference of one is noise. The derived rows used the layers as approved under the looser check; gemma4's now
leaves carryover and tuition unapproved, so it would fail questions 2 and 4 visibly and still answer 1 with the wrong
tiers. Claude's seconds in the first four setups were measured while Ollama was busy, so they may be a little high.

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

Claude-derived rules did as well as hand-written ones; gemma4 from Claude's layer left figures out (the match cap,
IRS limit and boot allowance) rather than getting them wrong. **Fully local, derivation lost**: gemma4 from its own
layer stated two wrong figures as fact on question 1, a match of $3,680 (key $3,220) and a maximum of $6,300 (key
$4,140), from the tiers the check passed. With supplied or Claude-derived rules no answer stated a key figure
wrongly. **Handbook only was close on Claude, not on gemma4**, which passed the rates to `tiered_match` wrongly,
answered a match of $32.20 and was slowest: the rules earn their keep most with the local model. With no tools,
Claude's one figure was the public IRS limit and its six hedged figures were what is typical elsewhere; both models
said they could not know Meridian's plan, as the prompt tells them to (a chatbot without that instruction was not
tested). Only the handbook gives question 5's $150 boot allowance: Claude found it; gemma4 did not this time, and on
question 1 it also left out the IRS limit and the vested percentage, two fewer figures than with rules only, which
may be noise. The 2024 edition was filtered out of derivation. Claude in handbook only read it and said it was
replaced on 1 January 2026; with rules the agents went straight to the scenarios. Figures no tool returned, mostly
the models' own arithmetic ("$920 a year in match"), are flagged in each answer file and not graded.

### Fewer tools

On 4 November 2026 gemma4:26b answered the five questions twice more, one run each (`measure.py run --setups
documents,fewer_tools`): once from the full rules-and-handbook server and once from the fewer-tools copy of it.

| Tool list | Tools | Key figures right | Wrong as fact | Not given | Verdicts right | Section cited | Calls | Seconds (mean) |
|---|---|---|---|---|---|---|---|---|
| Full | 22 | 13 of 17 | 0 | 4 | 4 of 4 | 3 of 5 | 17 | 13.5 |
| Limited | 7 | 13 of 17 | 0 | 4 | 3 of 4 | 5 of 5 | 11 | 12.8 |

The verdict the limited list missed: on question 1 it gave the match and the maximum without saying the match had
begun, and left out the cap and vesting the full list gave after calling `rules`; on question 3 it gave two weekly
pay figures the full list left out. The differences, and the one figure against the rules-and-handbook row above,
are noise: a shorter tool list neither helped nor hurt the local model.

The reader of the figures is Claude and may read an answer kindly. The key and the policies have the same author,
which says nothing about a real handbook, which is longer, contradicts itself and changes more often.
