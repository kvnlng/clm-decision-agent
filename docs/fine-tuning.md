# Training CLM for your scenario

As released, CLM is a general model, and on a specific set of routes it is often wrong: 8 of
20 hand-written messages on the example scenario. Training fits it to your routes in about a
minute.

## What is trained

Only CLM's two small heads (about 75 MB together). The large encoder is untouched, which is
why training takes seconds: every example is encoded once, and training only adjusts how
those vectors are projected so that each message lands nearest the description of its route.

Only the routing decision is trained. The scenario's other questions and the tool check stay
as released.

## Steps

**1. Get labelled examples.** Either source works, and they combine:

- `uv run python synthesize.py <scenario>` has the language model write example messages for
  each route, in eight styles, into `<scenario>.examples.jsonl`. Say who is writing and to what
  with `[synthesize] messages` in the scenario; the example's is "messages that staff might send
  to a help desk". Read some of the results. The file is plain text, one example per line, and
  you can edit it or add your own: `{"text": "<a message>", "route": "<a route's name>"}`
- Correct decisions in the test chat's review mode (`chat.py <scenario> --review`). Every route
  you correct is a real example.

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

| | Released | Trained |
|---|---|---|
| Held-back generated examples (61) | 31% | 98% |
| Hand-written test messages (20) | 8 | 16 |

Two earlier measurements on a similar problem (issue reports, 450 generated examples):

- Most of the gain arrived by 100 to 200 examples.
- A plain classifier on the encoder's vectors did about as well as trained heads. Trained
  heads were chosen because CLM then keeps doing every job through one model. `train.py`
  prints the classifier's accuracy beside its own so you can see the comparison on your data.

## Reading the numbers

- **Generated examples flatter the result.** The held-back examples were written by the same
  model in the same styles as the training examples. Believe a test on real or hand-written
  messages; keep such a set aside and never train on it.
- **Small tests move.** On 20 messages the trained result was 16 or 17 depending on the run.
- **Look at confidence as well as accuracy.** On the example, every wrong answer after
  training fell below the confidence threshold, so the assistant asked instead of acting. Run
  `evaluate.py` on your reviews to choose the threshold.

## When to train again

- **Whenever a route's description, a route's name or the routing question changes.** Trained
  heads fit the exact wording: in a test, rewording the options after training dropped
  accuracy below the untrained model. The server detects this, says so, and uses the released
  heads until you retrain.
- When you change the CLM build.
- When you have corrected more real messages.

## Limits

- Training runs on Apple Silicon only, with the local CLM backend. The CLM authors publish
  their own training script for GPU servers; this project does not use it.
- One example in five of the generated set is held back, chosen by a hash of its text, so the
  same examples are held back on every run. Reviewed routes are always trained on.
- After training, the "near-identical message" matching used to reuse reviews works on
  different vectors. A spot check looked fine at the default `match_threshold`; it has not
  been measured properly.
- Trained heads are made from your examples and reviews. Treat the folder with the same care
  as that text: see [data-handling.md](data-handling.md).
