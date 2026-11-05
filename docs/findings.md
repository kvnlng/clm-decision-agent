# What the traces showed

The CrowdStrike example is also the benchmark: runs were traced (`clm-server --trace`) before and after each change.
The numbers are in the [example's README](../examples/crowdstrike/README.md) and [serving.md](serving.md).

## Where the time goes

| Measurement | Result |
|---|---|
| CLM reading the texts, share of a `decide_many` call | 96–98% on the A100 and H100; all but 3 ms a call on the Mac |
| The tools (searching for journeys, holding seats) | 1–3% |
| H100 GPU busy, sampled every 5 seconds, day's average | 82%; the server used one CPU core, mostly waiting for the GPU |
| A second copy of the model on the same GPU | +4%: one copy already fills it |

The GPU is the limit, not the CPU; faster decisions come from encoding less, or better.

## What each change bought

The figures are decisions a second, `decide_many`'s own rate from the traces (the replay's scoreboard in the
example's README counts slightly more):

| Change | Before | After | Measured on |
|---|---|---|---|
| The tools share one HTTP client instead of making one per call (about 17 ms each on a cloud VM) | 23 | 55 | service desk, A100, 100 messages a call |
| Each message encoded once for all its questions (a shared start) | 55 | 72 | the same |
| Texts packed end to end in rows, no padding (42% of the tokens had been padding) | 10.9 | 12.5 | CrowdStrike, A100: before, the day's first 40,918 bookings; after, its first 3,000, rows of 4,096 tokens. Not the same work |
| Every request's options ranked in the questions' pass, not about 70 small passes a call | 12.5 | 17.4 | CrowdStrike, A100, first 3,000 bookings, rows of 4,096 |
| Rows of 2,048 tokens instead of 4,096 | 17.4 | 18.6 | the same |
| The H100 instead of the A100 | 18.6 | 45.8 | the same, rows of 2,048 |

The H100 decided the whole day at 50.1 a second.

- Packing alone bought little: with padding gone, ranking each request's options in its own small pass cost more than
  everything else.
- Rows of 1,024 or 2,048 tokens were fastest on both GPUs; on the A100, 4,096 was 6–7% slower and 8,192 25% slower,
  because attention covers the whole row.
- Before packing, a shared start took a second pass, slowing a single small decision, so it was used only with 1,024
  tokens or more to encode. In a packed row the start and the questions go in one pass, shared whenever texts in a row
  start alike.
- The day's busiest minute needs 10.7 decisions a second. The Mac (3.6) falls behind at the peaks by up to 4.5 minutes;
  one A100 keeps up; an H100 decides 4.8 times what the peak needs.

## What broke

| What | Where | Fix |
|---|---|---|
| The vector cache reached its 50,000 entries and evicted a text the same call was about to use; the run stopped with a `KeyError` | A100, first whole-day run, 28 minutes in, at 09:12 of the simulated day | The cache evicts only after the call has what it needs |
| Packed rows were right only with SDPA attention, which honours the mask keeping packed texts apart; eager attention would have added the mask to its scores, and every text would have read its neighbours | A review caught it before any wrong number was published | The encoder loads with SDPA and refuses anything else |
| Python 3.15 was picked; torch has no wheels for it yet | A fresh install | Python is capped below 3.15 |
| Packed and alone differ a little: encoded in a batch, a text's numbers come out very slightly different, and a decision on the edge between two routes can flip | Service desk, 318 messages: 2 flipped on the Mac, 1 on the A100, all near ties well below the confidence threshold ([serving.md](serving.md)). CrowdStrike: rows of different sizes gave 2,393 to 2,404 right of 3,000; which decisions changed was not traced. Heads trained again on the H100 gave 2,379 against the A100's 2,399; whether training or the GPU's rounding made the difference was not measured | None |

## What the measurements say about the decisions

| Finding | Figures |
|---|---|
| Many bookings go to a person | Threshold 0.6: 5% of the day's bookings got a wrong route not marked uncertain; 46% were marked uncertain, most answered rightly. `clm-evaluate --cases` puts the lowest threshold keeping the routes 90% right at 0.63: better-trained heads are the fix, not a lower threshold |
| Claude was more accurate, and on the Mac faster | 600 unseen cases: Sonnet 5.5 82% of routes right, Haiku 5.5 79%, CLM 75%; Sonnet took 267 ms a decision, CLM 300 ms. CLM costs nothing per decision, keeps the messages on the machine and marks most of its wrong answers uncertain |
| `check_call` is weak as a yes or no | A rebooking and a refund for the same passenger: it approved the right one and refused the wrong one for 114 of 200; compared with each other, 169 were told apart. `decide` was right for 162 and marks most of its mistakes uncertain |

From the local-document examples ([examples/noncompete](../examples/noncompete/README.md),
[examples/handbook](../examples/handbook/README.md)):

| Finding | Figures |
|---|---|
| Local documents made answers better sourced, not different | Non-compete: the rules carry the statutes, so with or without the document tools Claude and gemma4 gave every state's verdict rightly. On the Texas question Claude reached the same verdict either way, citing 13 opinions the tools returned against 9 |
| A large tool result can silently lose a local model its question | `scenario_narrow_to_enforce` returned about 1 MB; Ollama cuts the conversation to its context window, and gemma4 answered "please provide the question". At about 0.7 MB it answered, in over 7 minutes. Neither the scenario nor the agent warns yet |
| The CLM's two document judgements do not separate their labels | "Broader than needed" ordered 6 of 28 labelled pairs rightly, worse than chance; "departs" 28 of 40. No rule reads either |
| With a company's own documents, the missing answer is the gap, not a wrong one | Made-up handbook, no tools: Claude and gemma4 said they could not know (Claude gave 1 of 17 figures, the public IRS limit; gemma4 none). With supplied rules, 16 of 17 right and none of the 17 stated wrongly; the handbook added the one question no rule covers, and the policy text. One run each |
| Small results suit a local model | Handbook questions: gemma4 about as fast as Claude (13 to 17 s) and as accurate with the rules only (16 of 17 each); with the handbook 14 of 17 against Claude's 17 |
| Rules Claude derived from the handbook matched the hand-written ones | Claude's five tables and six plan figures, all 11, matched the hand-written ones over a grid of every bound; all 11 passed the first check (3 on the second try). Answering from them, Claude gave 17 of 17 key figures, as with the hand-written rules; gemma4 14 of 17, none wrong. A stricter check (a range's upper bound, whole words in quotes, figures in a rule's reason; dates, bare years and section references exempt) still passes all 11 |
| Fully local derivation let wrong figures through the check | gemma4 derived 9 of 11 that pass the stricter check (10 of 11 the first), but two are wrong: it read "100% of the first 3% ... 50% of the next 3%" as tiers of 100% and 50% of pay. Every figure is in the quote, so the check cannot tell a rate from a width: it catches figures not in the document, not figures in the wrong place, which is the person's part. The scripted approval passed them; gemma4 answering from its own layer stated two wrong figures as fact (a $3,680 match where the key says $3,220). gemma4 also needed thinking off: with thinking on, under a JSON schema, it ran past 100,000 tokens without answering |
| A second reading caught the misplaced figures when Claude read, only partly when a local model did | `clm-derive check` can have a second model derive each entry again without the first answer; a disagreement fails the check. On gemma4's layer, Claude caught both wrong tiers and flagged none of the 9 right entries; granite4:micro caught both but flagged 5 right entries (it read the rates as 3); qwen3:14b caught one and agreed with the other, making gemma4's own mistake, so a wrong figure passed both checks. gemma4 reading Claude's all-right layer raised 3 false alarms of 11. Agreement is not proof; every entry still needs a person. One run each ([a second reading](../examples/handbook/README.md#a-second-reading)) |
| The handbook alone, with the calculations, was nearly enough for Claude and not for gemma4 | No rules: Claude gave 16 of 17 and recognised the superseded 2024 edition as replaced; gemma4 10 of 17 and one wrong figure, passing the match's rates to the calculation wrongly |
