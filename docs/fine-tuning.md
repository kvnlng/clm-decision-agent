# Training CLM for your scenario

As released, CLM is a general model, and on a specific set of routes it is often wrong: 8 of
20 hand-written messages on the example scenario. Training fits it to your routes in about a
minute.

## What is trained

Only CLM's two small heads (about 75 MB together). The large encoder is untouched, which is
why training takes seconds: every example is encoded once, and training only adjusts how
those vectors are projected so that each message lands nearest the right answer.

The route is always trained. So is every question your examples label, and the choice among a
route's options where examples say which option was right; all of them in the same heads.
Questions no example labels, and the tool check, stay as released, and `train.py` reports how
far training moved their answers.

**Rare answers count as much as common ones.** If one example in ten is urgent, each urgent one
weighs nine times as much in training, and every result is reported per answer ("4 of 5"):
otherwise "never urgent" would score 90% and look like success.

## Steps

**1. Get labelled examples.** Either source works, and they combine:

- `uv run python synthesize.py <scenario>` has the language model write example messages for
  each route, in eight styles, into `<scenario>.examples.jsonl`. Say who is writing and to what
  with `[synthesize] messages` in the scenario; the example's is "messages that staff might send
  to a help desk". Read some of the results. The file is plain text, one example per line, and
  you can edit it or add your own: `{"text": "<a message>", "route": "<a route's name>"}`
- Correct decisions in the test chat's review mode (`chat.py <scenario> --review`). Every route
  you correct is a real example.

**Labelling the questions and options.** A line can also say the right answer to each question,
and for a route with options, which option is right. Only what a line labels is trained from it:

```json
{"text": "Need the registry by Friday", "route": "Data_Access",
 "answers": {"urgency": 2, "identifiable": true}}
{"text": "My flight was cancelled; I must be in Denver tonight", "route": "Rebook",
 "facts": ["The 18:05 to Denver has 3 free seats."],
 "options": [{"id": "UA1234", "text": "Departs 18:05, arrives 21:40"},
             {"id": "UA88", "text": "Departs tomorrow 07:10"}],
 "option": "UA1234"}
```

- A yes/no answer is `true` or `false`, a score is the level's number counting from 0, and a
  choice is the option's key.
- `facts` are what CLM read with the request, as your tools would find them. Every question
  hears a line's facts; corrections save them by tool, so each question hears only the facts
  its `[tools.inform]` settings give it, as when deciding.
- A line that names a question the scenario lacks, or an answer it cannot take, stops `train.py`
  with the line's number.

**2. Say where to save the result**, in the scenario:

```toml
[clm]
heads = "scenario.heads"
```

**3. Train.**

```bash
uv run python train.py <scenario>
```

It prints accuracy before and after on examples it held back, and how far the answers to your
other questions moved. If training did not improve on the held-back examples, nothing is
saved.

**4. Start the server.** It says at start-up whether it is using the trained heads.

Training is one step in a loop: write the scenario, generate examples, train, try messages in
the test chat and correct its decisions, evaluate, adjust, and train again.

## What it did on the example

On the 55 held-back generated examples, with the route, urgency and identifiable trained
together (each count is right of all with that answer):

| | Released | Trained |
|---|---|---|
| Route | 49% (balanced) | 96% (53 of 55) |
| Urgency: no deadline | 9 of 49 | 47 of 49 |
| Urgency: within weeks | 0 of 1 | 0 of 1 |
| Urgency: within days | 5 of 5 | 4 of 5 |
| Identifiable: no | 17 of 42 | 40 of 42 |
| Identifiable: yes | 13 of 13 | 13 of 13 |
| Hand-written test messages (17) through the server, route | | 15 |

- Training the questions did not cost the route: trained on the route alone it was 54 of 55
  and 15 of 17.
- "Within weeks" has two examples in all, one held back, so it cannot be learned from them.
  Label more of a rare answer before trusting it.
- The released heads said "urgent" and "identifiable" far too often; the questions were
  unusable before training and close to the route's accuracy after.

Two earlier measurements on a similar problem (issue reports, 450 generated examples):

- Most of the gain arrived by 100 to 200 examples.
- A plain classifier on the encoder's vectors did about as well as trained heads. Trained
  heads were chosen because CLM then keeps doing every job through one model. `train.py`
  prints the classifier's accuracy beside its own so you can see the comparison on your data.

## Reading the numbers

- **Generated examples flatter the result.** The held-back examples were written by the same
  model in the same styles as the training examples. Believe a test on real or hand-written
  messages; keep such a set aside and never train on it.
- **Small tests move.** With the earlier five-route example, the trained result on 20 messages
  was 16 or 17 depending on the run.
- **Look at confidence as well as accuracy.** A wrong route below the confidence threshold is
  marked uncertain and runs no action; a confident wrong one acts. Run `evaluate.py` on your
  corrections to choose the threshold.

## When to train again

- **Whenever a route's description, a route's name or the routing question changes.** Trained
  heads fit the exact wording: in a test, rewording the options after training dropped
  accuracy below the untrained model. The server detects this, says so, and uses the released
  heads until you retrain.
- **Whenever a trained question's wording changes**, or a route's options instructions if its
  ranking was trained. Questions that were not trained can change freely.
- When you change the CLM build.
- When you have corrected more real messages.

## Limits

- Training runs with either backend, on the machine you serve from. Heads trained with one
  backend are not used by the other ([models.md](models.md#heads-belong-to-one-backend)).
  Trained with the torch backend on a Mac, the example also reached 98% held-back and 15 of 17.
- One example in five of the generated set is held back, chosen by a hash of its text, so the
  same examples are held back on every run. Reviewed routes are always trained on.
- After training, the "near-identical message" matching used to reuse corrections works on
  different vectors. A spot check looked fine at the default `match_threshold`; it has not
  been measured properly.
- Trained heads are made from your examples and corrections. Treat the folder with the same care
  as that text: see [data-handling.md](data-handling.md).
