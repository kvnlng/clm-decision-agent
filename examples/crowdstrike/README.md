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
