# What the traces showed

The CrowdStrike example is also the benchmark: runs were traced (`server.py --trace`) before and after
each change. This page records where the time went, what each change bought and what broke. The numbers
behind it are in the [example's README](../examples/crowdstrike/README.md) and in
[serving.md](serving.md).

## Where the time goes

**Encoding is almost all of it.** CLM reading the texts took 96–98% of every `decide_many` call on the
A100 and the H100, and all but 3 ms a call on the Mac. The tools (searching for journeys, holding seats)
took 1–3%. Faster decisions come from encoding less, or encoding it better, not from the rest of the
server.

**The GPU is the limit, not the CPU.** On the H100 the GPU was busy 82% of the day on average (sampled every 5 seconds), while
the server used one CPU core, most of it waiting for the GPU. A second copy of the model on the same
GPU added 4%: one copy already fills it.

## What each change bought

Decisions a second, `decide_many`'s own rate from the traces (the replay's scoreboard, which the
example's README quotes, counts slightly more):

| Change | Before | After | Measured on |
|---|---|---|---|
| The tools share one HTTP client instead of making one per call (about 17 ms each on a cloud VM) | 23 | 55 | service desk, A100, 100 messages a call |
| Each message encoded once for all its questions (a shared start) | 55 | 72 | the same |
| Texts packed end to end in rows, with no padding (42% of the tokens had been padding) | 10.9 | 12.5 | CrowdStrike, A100: before, the day's first 40,918 bookings; after, its first 3,000, rows of 4,096 tokens. Not the same work |
| Every request's options ranked in the same pass as the questions, not about 70 small passes a call | 12.5 | 17.4 | CrowdStrike, A100, the first 3,000 bookings, rows of 4,096 |
| Rows of 2,048 tokens instead of 4,096 | 17.4 | 18.6 | the same |
| The H100 instead of the A100 | 18.6 | 45.8 | the same, rows of 2,048 |

The H100 decided the whole day at 50.1 a second.

- Packing on its own bought little because the time had moved: once padding was gone, ranking each
  request's options in its own small pass cost more than everything else. The trace showed it.
- Rows of 1,024 or 2,048 tokens were fastest on both GPUs; 4,096 was 6–7% slower and 8,192 25% slower
  on the A100, because attention covers the whole row.
- Before packing, a shared start took a second pass, which made a single small decision slower; it was
  used only with 1,024 tokens or more to encode. In a packed row the start and the questions go in one
  pass, so it is shared whenever texts in a row start alike.
- **What the speed is for:** the day's busiest minute needs 10.7 decisions a second. The Mac (3.6)
  falls behind at the peaks, by up to 4.5 minutes; one A100 keeps up, and an H100 decides 4.8 times as
  many as the peak needs.

## What broke

- **The cache dropped a text it still needed.** On the A100's first whole-day run, the vector cache
  reached its 50,000 entries and evicted a text the same call was about to use; the run stopped with a
  `KeyError` 28 minutes in, at 09:12 of the simulated day. The cache now evicts only after the call has what it needs.
- **The packed rows were right only with one attention implementation.** The mask that keeps packed
  texts apart is honoured by SDPA; with eager attention, transformers would have added it rather than
  applied it, and every text would have read its neighbours. A review caught it before any wrong
  number was published. The encoder now loads with SDPA and refuses anything else.
- **A fresh install picked Python 3.15**, which torch has no wheels for yet. Python is capped below
  3.15.
- **Packed and alone differ a little.** Encoded in a batch, a text's numbers come out very slightly
  different, and a decision on the edge between two routes can flip. On the service desk's 318 messages,
  2 did on the Mac and 1 on the A100, all near ties well below the confidence threshold ([serving.md](serving.md)). On the CrowdStrike
  example, rows of different sizes gave 2,393 to 2,404 right of 3,000; which decisions changed was not
  traced. Heads trained again on the H100 gave 2,379, against the A100's 2,399; whether training or the
  GPU's rounding made the difference was not measured.

## What the measurements say about the decisions

- **Many bookings go to a person.** With the threshold at 0.6, 5% of the day's bookings got a wrong
  route not marked uncertain, but 46% were marked uncertain, most of them answered rightly.
  `evaluate.py --cases` puts the lowest threshold that keeps the routes 90% right at 0.63, so lowering
  it is not the fix; better-trained heads are.
- **Claude was more accurate, and on the Mac faster.** On 600 unseen cases Sonnet 5.5 got 82% of routes
  right, Haiku 5.5 79%, CLM 75%, at 267 against 300 ms a decision. CLM costs nothing per decision, keeps
  the messages on the machine and marks most of its wrong answers uncertain.
- **`check_call` is weak as a yes or no.** Asked about both a rebooking and a refund for the same
  passenger, it approved the right one and refused the wrong one for 114 of 200; compared with each
  other the two were told apart in 169. `decide` itself was right for 162, and marks most of its
  mistakes uncertain.
