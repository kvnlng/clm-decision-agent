# The CrowdStrike day

On 19 July 2024 a faulty CrowdStrike update crashed Windows machines around the world. US
airlines cancelled 3,548 flights that day, Delta alone 1,325, and hundreds of thousands of
passengers needed a new plan at once. This example replays that day: a decision server has to
decide, for every passenger who writes in, how to help them.

The flights are real. Everything else (seats, passengers, fares, messages and the rebooking
policy) is made up for the demo, generated from a seed so every run is the same.

## The flights

`flights.csv.gz` holds every US flight of 19 July 2024, and those of 20 July leaving before noon,
with times as minutes from 19 July 00:00 UTC. `airports.csv` gives each airport's UTC offset in
minutes and its city, for showing local times and places.

Source: Bureau of Transportation Statistics, "Reporting Carrier On-Time Performance (1987-present)",
July 2024. A U.S. Government Work, in the public domain.

To rebuild it, download the July 2024 file from
<https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr> (the
"Prezipped File" for 2024, July), unzip it, and run:

```bash
uv run python examples/crowdstrike/make_extract.py <the unzipped CSV>
```

BTS gives local times without time zones. The script works out each airport's offset from the
schedule itself, and drops the 8 flights whose own schedule contradicts itself.

## What the world knows

The search for alternatives uses each flight's real departure and arrival, delays included, so the
options it offers know how the day turned out. The policy's right answers use the same knowledge.

## The scenario

`scenario.toml` decides for one booking at a time, or many at once with `decide_many`. Each request
carries the booking in its context: `{"booking": "<id>", "now": <minute>}` (without `now`, the moment
the passenger wrote). The routes are the six answers of `policy.md`, whose sections are their
descriptions word for word.

- **Before the route is chosen**, `situation` states the passenger's position in plain sentences:
  the booking, what happened to it, the fare, and the soonest journeys still possible (our own
  flights today, then a partner's, then ours tomorrow morning). The route is decided from the
  message and these facts.
- **On the rebooking routes**, `alternatives` offers up to three journeys as the options, soonest
  first, and CLM chooses one.
- **The actions** hold the whole party's seats on the chosen journey, book a hotel for an
  overnight, refund, and draft the reply. A hold that loses the race for the last seats fails,
  naming the full flight, and then no hotel is booked and no "we have booked you" reply is drafted.
- **The questions**: how firm a time to arrive by is, whether someone vulnerable travels, rail or
  air, and how upset the passenger is. Flight facts are given to none of them.

The tools see only what a booking record holds: the flights, the party and the fare. What the
passenger wants, any time they must arrive by and how they feel reach the server only through
their message.

Two calls worth knowing about:

- **Times are compared by the tool, not by CLM.** For 5% of the day's passengers the right answer
  turns on whether a flight lands before a time they must arrive by. CLM compares meanings; it does
  not do arithmetic. So `situation` finds clock times in the message ("by 18:00", "09:00 on 20 July")
  and says of each journey whether it lands before or after them. Whether the time is a must is
  CLM's judgment. The parser reads the forms the generated messages use; real messages would need
  more.
- **Choosing the option is mostly arithmetic here.** Under this policy the right journey is always
  the one that lands first, and its option says so. A high option score shows CLM reads that
  sentence, not that it weighs journeys.

## Training and checking

```bash
uv run python examples/crowdstrike/write_examples.py      # 1,500 examples and 600 check cases
uv run python train.py examples/crowdstrike/scenario.toml
uv run python examples/crowdstrike/check.py examples/crowdstrike/scenario.toml
```

`write_examples.py` writes 250 examples of each route from seed 1's day, with the facts and
options the scenario's own tools give, and 100 check cases of each route from seed 0's day, the
demo's. A fifth of the message pool's lines, chosen by a hash of the text, is kept out of training
and used only for the check cases, so the check is on wording CLM never saw. `check.py` runs them
through the server's own steps, without actions, and reports every result per class.

### What it did

On an Apple M5 Max (64 GB, MLX, the 8-bit CLM), trained on the 1,500 examples, checked on the 600
unseen cases (each count is right of all with that answer):

| | Released | Trained |
|---|---|---|
| Route | 100 of 600 (17%): Overnight for everyone | 448 of 600 (75%) |
| Rebook | 0 of 100 | 81 of 100 |
| Rebook_Partner | 0 of 100 | 85 of 100 |
| Overnight | 100 of 100 | 69 of 100 |
| Refund | 0 of 100 | 81 of 100 |
| Agent | 0 of 100 | 62 of 100 |
| No_Action | 0 of 100 | 70 of 100 |
| Wrong routes marked uncertain (no action runs) | 10 of 500 | 119 of 152 |
| Option, where the route was right | 67 of 100 | 235 of 235 |
| Deadline: none / would like / must | 60, 113, 0 of 243, 134, 95 | 198, 116, 82 |
| Vulnerable: no / yes | 335, 56 of 539, 61 | 535, 58 |
| Travel mode: air / either | 0, 80 of 377, 95 | 375, 88 |
| Upset: calm / annoyed / angry | 0, 213, 22 of 245, 257, 98 | 195, 159, 49 |

- **Most wrong routes are caught.** 119 of the 152 wrong routes fell below the confidence
  threshold, so no action would run on them; 33 (6% of all) were confidently wrong.
- **Agent and Overnight are the hardest routes.** Agent covers two different cases: a message too
  vague to act on, and a firm deadline nothing meets on a fare that can't be refunded. Overnight
  turns on there being nothing today, which several facts together say.
- **On its own held-back examples training gave 76%, where a plain classifier on the same vectors
  gave 89%.** The heads have room to improve.
- **Rail is not trained.** Only 180 of the day's 99,806 bookings prefer the train (it is offered
  only within the north-east corridor), so the examples hold none.

## The demo

```bash
uv run python examples/crowdstrike/replay_day.py examples/crowdstrike/scenario.toml --hub ATL
uv run --group demo python examples/crowdstrike/compare_models.py examples/crowdstrike/scenario.toml
```

`replay_day.py` plays the coordinator. It starts the decision server (with `--trace`), sends the day's
messages through `decide_many` a simulated minute at a time, and redraws a scoreboard: decisions a second,
how many were automatic and how many went to a person (and why), seat holds made and lost, and accuracy.
It keeps its own copy of the seats and makes the same holds the server reports, so every booking is
judged against the seats left when it was decided. `--url` uses a server already running over HTTP.

`compare_models.py` gives CLM, a local language model and Claude the same 600 check cases: the message,
the situation's facts, every rebooking route's journeys (fetched beforehand) and the policy's route
descriptions. The language models answer with a route and a journey and call no tools. Claude runs at
effort `low` with JSON output and the policy cached; a refusal would count as wrong. The Claude models
need credentials (`ant auth login` or `ANTHROPIC_API_KEY`) and are skipped without them.

### CLM against language models

Apple M5 Max (64 GB); CLM and Qwen3-4B on MLX on that machine, Claude through the Anthropic API with 8
calls at once; prices of 22 October 2026. Day-weighted accuracy weights each route by its share of the
day's right answers.

| | Claude Sonnet 5.5 | Claude Haiku 5.5 | CLM (trained heads) | Qwen3-4B (local) |
|---|---|---|---|---|
| Right route, of 600 | **494 (82%)** | 473 (79%) | 448 (75%) | 360 (60%) |
| Day-weighted accuracy | **84%** | 81% | 76% | 61% |
| Confidently wrong (an action runs on it) | 106 | 127 | **33** | 240 |
| Time per decision | 267 ms | **266 ms** | 300 ms | 390 ms |
| Cost of the whole day (99,806 decisions) | $158 | $15 | **$0** | **$0** |
| Where the messages go | Anthropic | Anthropic | **this machine** | **this machine** |

Per route (right of 100):

| | Rebook | Partner | Overnight | Refund | Agent | No_Action |
|---|---|---|---|---|---|---|
| Sonnet 5.5 | 100 | 97 | 93 | 50 | 60 | 94 |
| Haiku 5.5 | 100 | 94 | 97 | 43 | 54 | 85 |
| CLM | 81 | 85 | 69 | 81 | 62 | 70 |
| Qwen3-4B | 95 | 91 | 89 | 75 | 2 | 8 |

- **On this Mac, Claude was both more accurate and faster.** Claude's calls take about 2 seconds each,
  but 8 run at once; CLM's time is almost all encoding on the Mac's GPU. More concurrent calls would make
  Claude faster still, up to its rate limits. Whether CLM is faster on a data-centre GPU is measured on
  the A100 below.
- **CLM costs nothing per decision and keeps the messages on the machine.** Haiku would cost about $15
  for the day, Sonnet about $158.
- **CLM knows when it is unsure.** 119 of its 152 wrong routes were marked uncertain, so no action ran
  and they went to a person; the language models gave no such signal, so every wrong answer acted. The
  price is that CLM hands many right answers to a person too (see the replay below).
- **The language models do worst on Refund and Agent**, the routes that turn on what the passenger
  wants (no longer to travel, or a message too vague to act on) or on a time nothing meets. CLM, trained
  on the policy's examples, does better there and worse on the routes the models find easy.
- The comparison cost about $1.05 in API calls (600 cases on each Claude model, at about 1,370 input
  tokens a case, 852 of them cached).

### The day at Atlanta, replayed

Every booking starting at Atlanta, 9,704 bookings and 15,503 passengers, from 04:10 UTC on 19 July to
07:36 UTC on 20 July, on the Apple M5 Max (MLX, trained heads):

| | |
|---|---|
| Time | about 45 minutes, 3.6 decisions a second |
| Right route | 7,529 of 9,704 (78%) |
| Confidently wrong (an action ran on it) | 465 (4.8%) |
| Automatic | 5,050 (52%) |
| To a person | 4,654 (48%): uncertain 4,142, Agent route 1,263, a hold lost 28 (a booking can have two reasons) |
| Seat holds | 2,868 made, 28 lost to a passenger decided just before |
| The coordinator's seats against the server's | the same throughout |

Per route: Rebook 3,860 of 4,558, Rebook_Partner 73 of 94, Overnight 282 of 433, Refund 1,240 of 1,633,
Agent 812 of 1,454, No_Action 1,262 of 1,532.

- **Encoding is all the time.** The trace puts 100% of each `decide_many` call in CLM's encoding; the
  tools (searching for alternatives, holding seats) take about 3 ms a call.
- **Too many go to a person.** The confidence threshold of 0.6 keeps confidently wrong answers to 5%,
  but marks 43% of all bookings uncertain, most of them answered rightly. A lower threshold, chosen with
  `evaluate.py` on corrections, or better-trained heads would move that balance.
- Seats lost to a race are real: 28 passengers were offered a journey whose last seats went to someone
  decided a moment earlier, and their hold failed visibly, as it should.
