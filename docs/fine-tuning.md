# Training CLM for your scenario

As released, CLM is general and often wrong on a specific set of routes: 8 of 20 hand-written messages on the example
scenario. Training fits it to your routes in about a minute.

## What is trained

Only CLM's two small heads (about 75 MB together). The large encoder is untouched: each example is encoded once, and
training adjusts how those vectors are projected so each message lands nearest the right answer.

The route is always trained, with every question your examples label and the choice among a route's options where
examples say which was right, all in the same heads. Unlabelled questions and the tool check stay as released;
`clm-train` reports how far their answers moved.

**Rare answers count as much as common ones.** If one example in ten is urgent, each urgent one weighs nine times as
much, and every result is reported per answer ("4 of 5"): otherwise "never urgent" would score 90%.

## Steps

**1. Get labelled examples.** Either source works, and they combine:

- `uv run clm-synthesize <scenario>` has the language model write messages for each route, in eight styles, into
  `<scenario>.examples.jsonl`. `[synthesize] messages` says who is writing and to what; the example's is "messages
  that staff might send to a help desk". Read some of the results. Edit the file freely, or add lines:
  `{"text": "<a message>", "route": "<a route's name>"}`
- Correct decisions in review mode: start the server with `--review` in the MCP Inspector
  (`npx -y @modelcontextprotocol/inspector@2.13.0 uv run clm-server <scenario> --review`), call `decide` with a
  message, then `correct` with the same text and the right route. Every corrected route is a real example.

**Labelling the questions and options.** A line can also give each question's right answer and, on a route with
options, the right option. Only what a line labels is trained from it:

```json
{"text": "Need the registry by Friday", "route": "Data_Access",
 "answers": {"urgency": 2, "identifiable": true}}
{"text": "My flight was cancelled; I must be in Denver tonight", "route": "Rebook",
 "facts": ["The 18:05 to Denver has 3 free seats."],
 "options": [{"id": "UA1234", "text": "Departs 18:05, arrives 21:40"},
             {"id": "UA88", "text": "Departs tomorrow 07:10"}],
 "option": "UA1234"}
```

- A yes/no answer is `true` or `false`, a score is the level's number counting from 0, and a choice is the option's key.
- `facts` are what CLM read with the request, as your tools would find them. Given as a list, every question hears
  them. Given by tool, as corrections save them (`{"situation": ["..."], "lookup": ["..."]}`), each question hears only
  the facts its `[tools.inform]` settings give it, and the facts of `[routing] inform` tools are added to the text the
  route is trained on, as when deciding.
- A line naming a question the scenario lacks, or an answer it cannot take, stops `clm-train` with the line's number.

**2. Say where to save the result**, in the scenario:

```toml
[clm]
heads = "scenario.heads"
```

**3. Train.**

```bash
uv run clm-train <scenario>
```

It prints accuracy before and after on held-back examples, and how far your other questions' answers moved. If the
held-back examples did not improve, nothing is saved.

**4. Start the server.** It says at start-up whether it is using the trained heads.

Then loop: generate examples, train, try messages in the Inspector and correct them, evaluate, adjust, train again.

## What it did on the example

On the 55 held-back generated examples, with the route, urgency and identifiable trained together (each count is right
of all with that answer):

| | Released | Trained |
|---|---|---|
| Route | 49% (balanced) | 96% (53 of 55) |
| Urgency: no deadline | 9 of 49 | 47 of 49 |
| Urgency: within weeks | 0 of 1 | 0 of 1 |
| Urgency: within days | 5 of 5 | 4 of 5 |
| Identifiable: no | 17 of 42 | 40 of 42 |
| Identifiable: yes | 13 of 13 | 13 of 13 |
| Hand-written test messages (17) through the server, route | | 15 |

- Training the questions did not cost the route: trained on the route alone it was 54 of 55 and 15 of 17.
- "Within weeks" has two examples in all, one held back, so it cannot be learned from them. Label more of a rare answer
  before trusting it.
- The released heads said "urgent" and "identifiable" far too often; the questions were unusable before training and
  close to the route's accuracy after.

Two earlier measurements on a similar problem (issue reports, 450 generated examples):

- Most of the gain arrived by 100 to 200 examples.
- A plain classifier on the encoder's vectors did about as well as trained heads, which were chosen so CLM does every
  job through one model. `clm-train` prints the classifier's accuracy beside its own.

## Reading the numbers

- **Generated examples flatter the result.** The held-back ones come from the same model and styles as the training
  ones. Believe a test on real or hand-written messages; keep such a set aside and never train on it.
- **Small tests move.** With the earlier five-route example, the trained result on 20 messages was 16 or 17 depending on
  the run.
- **Look at confidence as well as accuracy.** A wrong route below the confidence threshold is marked uncertain and runs
  no action; a confident wrong one acts. Run `clm-evaluate` on your corrections to choose the threshold.

## When to train again

- **Whenever a route's description, a route's name or the routing question changes.** Trained heads fit the exact
  wording: in a test, rewording the options after training dropped accuracy below the untrained model. The server
  detects this, says so, and uses the released heads until you retrain.
- **Whenever a trained question's wording changes**, or a route's options instructions if its ranking was trained.
  Untrained questions can change freely.
- When you change the CLM build.
- When you have corrected more real messages.

## Limits

- Training runs with either backend, on the machine you serve from. Heads trained with one backend are not used by the
  other ([models.md](models.md#heads-belong-to-one-backend)). Trained with the torch backend on a Mac, the example also
  reached 98% held-back and 15 of 17.
- One example in five of the generated set is held back, chosen by a hash of its text, so the same examples are held
  back on every run. Reviewed routes are always trained on.
- After training, the "near-identical message" matching used to reuse corrections works on different vectors. A spot
  check looked fine at the default `match_threshold`; it has not been measured properly.
- Trained heads are made from your examples and corrections. Treat the folder with the same care as that text: see
  [data-handling.md](data-handling.md).
