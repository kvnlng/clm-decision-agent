# Building a scenario, step by step

This goes through building a scenario from nothing, with the CrowdStrike example
([examples/crowdstrike](../examples/crowdstrike/README.md)) as the worked case. Each step says what
to do, what the example did, and where the details are. [scenario.md](scenario.md) explains every
setting.

## 1. Start a folder

```bash
uv run python new_scenario.py my_scenario
```

This writes a small scenario that works straight away, for a shop's customer messages:

- `scenario.toml`, with three routes, a question, a rule, an informing tool and an acting tool;
- `my_scenario_tools.py`, with the two tools;
- `scenario.cases.jsonl`, a few labelled cases to check it on.

The `[clm]` block suits the machine: MLX on a Mac with Apple Silicon, torch elsewhere. Run
`evaluate.py my_scenario/scenario.toml --cases my_scenario/scenario.cases.jsonl` to see it decide
before you change anything. Then edit each part into your own.

## 2. Write the policy as routes

A route is one thing the server can decide to do, with a description CLM compares the message
against. Write the descriptions as you would brief a person: what is true of the request when this
route is right. CLM always picks one route, so keep one for everything else.

The CrowdStrike example has a written policy, `policy.md`, with six answers. Each answer's section
is its route's description word for word, and a test checks that they match, so the policy and the
scenario cannot drift apart.

## 3. Give the decision what it needs to know: informing tools

**Information reaches a decision only through tools.** An informing tool returns plain sentences,
and the tool, not CLM, does the arithmetic and the lookups: "the connection will be missed", not
two times for CLM to compare. CLM compares meanings; it does not calculate.

- Facts that decide the route go under `[routing] inform`: the example's `situation` says what
  happened to the booking and which journeys are still possible, and the route is chosen with them.
- A choice among things that exist only at that moment is a route's **options**: the example's
  `alternatives` offers up to three journeys and CLM picks one.
- Say which questions hear which facts (`[tools.inform.<tool>] questions`): flight facts would sway
  how upset a passenger seems, so in the example no question hears them.

Tools find their records from the request's `context`, which CLM never reads: the example sends
`{"booking": "<id>", "now": <minute>}`.

## 4. Act on it: acting tools

An acting tool runs as soon as the decision is made, on the routes you list, unless the route is
uncertain. Put every safeguard in the tool: the example's `hold_seats` fails, naming the full flight,
when another passenger took the last seats a moment earlier, and the reply says so.

**Build each tool with a mock backend first.** The starter's tools read `[tools.settings] backend`:
`"mock"` works now, and `"live"` is where you call your real systems. The scenario can be trained,
checked and demonstrated before anything real is connected. The service desk example does the same
with its tracker: `tracker_url` points at a stand-in (`mock_tracker.py`) or at a real Forgejo
([its README](../examples/service_desk/README.md)).

## 5. Questions and rules

Questions are what the decision needs to know besides the route: how urgent, whether someone
vulnerable is involved. Rules turn answers into instructions for the reply ("upset >= 1.5: apologise
first"). The example asks four questions and has three rules.

## 6. Examples to train on, and cases to check on

Training needs labelled examples of each route, ideally with the right answer to each question and
the right option. Checking needs **cases whose wording training never saw**, or the check measures
memory.

- Without data, `synthesize.py` has a language model write examples for each route.
- The example generated its own: a local model wrote a pool of message lines, and
  `write_examples.py` assembled 1,500 examples from one simulated day and 600 check cases from
  another, keeping a fifth of the pool's lines for the check cases only.
- Label rare answers too: training balances them, and every report is per class, so "always no"
  cannot pass as success.

## 7. Train and check

```bash
uv run python train.py my_scenario/scenario.toml
uv run python evaluate.py my_scenario/scenario.toml --cases my_scenario/scenario.cases.jsonl
```

`evaluate.py --cases` decides every case through the server's own steps, with your tools but no
acting tool, and reports each route, option and answer per class. It also shows accuracy by CLM's
confidence and suggests the lowest `confidence_threshold` at which the routes kept are 90% right.
On the example, trained heads took the route from 17% to 75% on unseen cases; the suggested
threshold was 0.63, against the 0.6 it uses.

Once people review real decisions, their corrections become training data too
([fine-tuning.md](fine-tuning.md)); `evaluate.py` without `--cases` replays them.

## 8. Serve it and measure

```bash
uv run python server.py my_scenario/scenario.toml --trace runs/my.trace.jsonl
```

`--trace` records where each call's time went; `trace_report.py` sums it up
([serving.md](serving.md)). The example's `replay_day.py` plays a coordinator sending a whole day
through `decide_many` and keeps a scoreboard; [findings.md](findings.md) records what its traces
showed.
