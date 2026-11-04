# What the traces showed

The CrowdStrike example is also the benchmark: runs were traced (`clm-server --trace`) before and after each change. This
page records where the time went, what each change bought and what broke. The numbers behind it are in the
[example's README](../examples/crowdstrike/README.md) and [serving.md](serving.md).

## Where the time goes

**Encoding is almost all of it.** CLM reading the texts took 96–98% of every `decide_many` call on the A100 and the H100,
and all but 3 ms a call on the Mac. The tools (searching for journeys, holding seats) took 1–3%. Faster decisions come
from encoding less, or better, not from the rest of the server.

**The GPU is the limit, not the CPU.** On the H100 the GPU was busy 82% of the day on average (sampled every 5 seconds),
while the server used one CPU core, mostly waiting for the GPU. A second copy of the model on the same GPU added 4%: one
copy already fills it.

## What each change bought

Decisions a second, `decide_many`'s own rate from the traces (the replay's scoreboard, quoted in the example's README,
counts slightly more):

| Change | Before | After | Measured on |
|---|---|---|---|
| The tools share one HTTP client instead of making one per call (about 17 ms each on a cloud VM) | 23 | 55 | service desk, A100, 100 messages a call |
| Each message encoded once for all its questions (a shared start) | 55 | 72 | the same |
| Texts packed end to end in rows, with no padding (42% of the tokens had been padding) | 10.9 | 12.5 | CrowdStrike, A100: before, the day's first 40,918 bookings; after, its first 3,000, rows of 4,096 tokens. Not the same work |
| Every request's options ranked in the questions' pass, not about 70 small passes a call | 12.5 | 17.4 | CrowdStrike, A100, the first 3,000 bookings, rows of 4,096 |
| Rows of 2,048 tokens instead of 4,096 | 17.4 | 18.6 | the same |
| The H100 instead of the A100 | 18.6 | 45.8 | the same, rows of 2,048 |

The H100 decided the whole day at 50.1 a second.

- Packing alone bought little because the time had moved: once padding was gone, ranking each request's options in its
  own small pass cost more than everything else. The trace showed it.
- Rows of 1,024 or 2,048 tokens were fastest on both GPUs; 4,096 was 6–7% slower and 8,192 25% slower on the A100, because
  attention covers the whole row.
- Before packing, a shared start took a second pass, which slowed a single small decision; it was used only with 1,024
  tokens or more to encode. In a packed row the start and the questions go in one pass, so it is shared whenever texts in
  a row start alike.
- **What the speed is for:** the day's busiest minute needs 10.7 decisions a second. The Mac (3.6) falls behind at the
  peaks, by up to 4.5 minutes; one A100 keeps up, and an H100 decides 4.8 times as many as the peak needs.

## What broke

- **The cache dropped a text it still needed.** On the A100's first whole-day run, the vector cache reached its 50,000
  entries and evicted a text the same call was about to use; the run stopped with a `KeyError` 28 minutes in, at 09:12 of
  the simulated day. The cache now evicts only after the call has what it needs.
- **The packed rows were right only with one attention implementation.** SDPA honours the mask that keeps packed texts
  apart; with eager attention, transformers would have added it rather than applied it, and every text would have read
  its neighbours. A review caught it before any wrong number was published. The encoder now loads with SDPA and refuses
  anything else.
- **A fresh install picked Python 3.15**, which torch has no wheels for yet. Python is capped below 3.15.
- **Packed and alone differ a little.** Encoded in a batch, a text's numbers come out very slightly different, and a
  decision on the edge between two routes can flip. On the service desk's 318 messages, 2 did on the Mac and 1 on the
  A100, all near ties well below the confidence threshold ([serving.md](serving.md)). On the CrowdStrike example, rows of
  different sizes gave 2,393 to 2,404 right of 3,000; which decisions changed was not traced. Heads trained again on the
  H100 gave 2,379, against the A100's 2,399; whether training or the GPU's rounding made the difference was not measured.

## What the measurements say about the decisions

- **Many bookings go to a person.** With the threshold at 0.6, 5% of the day's bookings got a wrong route not marked
  uncertain, but 46% were marked uncertain, most of them answered rightly. `clm-evaluate --cases` puts the lowest
  threshold that keeps the routes 90% right at 0.63, so lowering it is not the fix; better-trained heads are.
- **Claude was more accurate, and on the Mac faster.** On 600 unseen cases Sonnet 5.5 got 82% of routes right, Haiku 5.5
  79%, CLM 75%, at 267 against 300 ms a decision. CLM costs nothing per decision, keeps the messages on the machine and
  marks most of its wrong answers uncertain.
- **`check_call` is weak as a yes or no.** Asked about both a rebooking and a refund for the same passenger, it approved
  the right one and refused the wrong one for 114 of 200; compared with each other the two were told apart in 169.
  `decide` itself was right for 162, and marks most of its mistakes uncertain.
- **Local documents made answers better sourced, not different.** In the non-compete example the rules carry the
  statutes, so with or without the document tools both Claude and gemma4 gave every state's verdict rightly. On the
  Texas question Claude reached the same verdict either way, citing 13 opinions the tools returned against 9.
  ([examples/noncompete](../examples/noncompete/README.md))
- **A large tool result can silently lose a local model its question.** `scenario_narrow_to_enforce` returned
  about 1 MB; Ollama cuts the conversation to its context window, and gemma4 answered "please provide the
  question". At about 0.7 MB it answered, in over 7 minutes. Neither the scenario nor the agent warns yet.
- **The CLM's two document judgements do not separate their labels.** "Broader than needed" ordered 6 of 28
  labelled pairs rightly, worse than chance; "departs" 28 of 40. No rule reads either.
- **With a company's own documents, the missing answer is the gap, not a wrong one.** Asked about a made-up
  handbook with no tools, Claude and gemma4 said they could not know (Claude gave 1 of 17 figures, the public IRS
  limit; gemma4 none) rather than invent figures; with supplied rules no answer stated one of the 17 key figures
  wrongly. With the rules, 16 of 17 figures were right; the handbook added the one question no rule covers, and the
  policy text. One run each. ([examples/handbook](../examples/handbook/README.md))
- **Small results suit a local model.** On the handbook's questions gemma4 was about as fast as Claude (13 to 17 s)
  and as accurate with the rules only (16 of 17 each); with the handbook it gave 14 of 17 against Claude's 17.
- **Rules derived by Claude from the handbook matched the hand-written ones.** A rule is supplied, or derived from
  the documents and checked before a person approves it. Claude derived the handbook's five tables and six plan
  figures; all 11 passed the mechanical check (3 on the second try) and all 11 matched the hand-written tables and
  figures over a grid of every bound. Answering from them, Claude gave 17 of 17 key figures, as with the hand-written
  rules; gemma4 gave 14 of 17, none wrong.
- **Fully local derivation let wrong figures through the check.** gemma4 derived 10 of 11 that passed, but two of
  them were wrong: it read "100% of the first 3% ... 50% of the next 3%" as tiers of 100% and 50% of pay. Every
  figure is in the quote, so the mechanical check cannot tell a rate from a width; the scripted approval standing in
  for a person passed them, and gemma4 answering from its own layer stated two wrong figures as fact (a $3,680 match
  where the key says $3,220). The check catches figures that are not in the document, not figures put in the wrong
  place: that is the person's part. gemma4 also needed thinking off (with it, under a JSON schema, it ran past
  100,000 tokens without answering).
- **The handbook alone, with the calculations, was nearly enough for Claude and not for gemma4.** With no rules,
  Claude gave 16 of 17 and recognised the superseded 2024 edition as replaced; gemma4 gave 10 of 17 and one wrong
  figure, passing the match's rates to the calculation wrongly.
