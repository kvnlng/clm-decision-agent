# The CrowdStrike day

On 19 July 2024 a faulty CrowdStrike update crashed Windows machines worldwide; US airlines cancelled
3,548 flights that day, Delta alone 1,325. A decision server replays the day, deciding, for every
passenger who writes in, how to help them. The flights are real; seats, passengers, fares, messages and
the policy are made up from a seed, so every run is the same.

## The flights

`flights.csv.gz` holds every US flight of 19 July 2024 and those of 20 July leaving before noon, times
in minutes from 19 July 00:00 UTC; `airports.csv` holds each airport's UTC offset and city. Source: Bureau of
Transportation Statistics, "Reporting Carrier On-Time Performance (1987-present)", July 2024, a U.S.
Government Work in the public domain. To rebuild, download the July 2024 "Prezipped File" from
<https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr>, unzip it, and run:

```bash
uv run python examples/crowdstrike/make_extract.py <the unzipped CSV>
```

BTS gives local times without zones; the script works out offsets from the schedule itself and drops
the 8 flights whose schedule contradicts itself. Alternatives and right answers use the real departures
and arrivals, delays included.

## The scenario

`scenario.toml` decides one booking at a time, or many with `decide_many`; each request's context is
`{"booking": "<id>", "now": <minute>}`; without `now`, the moment is when the passenger wrote. The routes are
the six answers of `policy.md`, whose sections are their descriptions word for word.

| Step | What it does |
|---|---|
| `situation`, before the route | Writes plain sentences: the booking, what happened to it, the fare and the soonest journeys still possible, our own flights today first, then a partner's, then ours tomorrow morning. The route is decided from the message and these facts. |
| `alternatives`, on the rebooking routes | Up to three journeys as the options, soonest first; CLM chooses one. |
| Actions | Hold the party's seats, book a hotel for an overnight, refund, draft the reply. A hold that loses the race for the last seats fails, naming the full flight; after that there is no hotel and no "we have booked you" reply. |
| Questions | How firm a deadline is, whether someone vulnerable travels, rail or air, how upset the passenger is. None hears flight facts. |

The tools see only the booking record; what the passenger wants, any deadline and how they feel reach
the server only through the message. Two caveats. First, times are compared by the tool, not CLM: 5% of
answers turn on whether a flight lands before a deadline, so `situation` finds clock times in the
message ("by 18:00", "09:00 on 20 July") and says whether each journey lands before or after. How firm
the time is stays CLM's judgement; the parser reads only the forms the generated messages use. Second,
the right journey is always the one that lands first, and its option says so: a high option score shows
CLM reads that sentence, not that it weighs journeys.

## Training and checking

```bash
uv run python examples/crowdstrike/write_examples.py      # 1,500 examples and 600 check cases
uv run clm-train examples/crowdstrike/scenario.toml
uv run clm-evaluate examples/crowdstrike/scenario.toml --cases examples/crowdstrike/check.cases.jsonl
```

`write_examples.py` writes 250 examples per route from seed 1's day and 100 check cases per route from
seed 0's (the demo's), with the facts and options the scenario's own tools give. A fifth of the message
pool, chosen by a hash of the text, is kept for the check cases only, so the check is on wording CLM
never saw. `clm-evaluate --cases` runs them through the server's own steps without actions, reports
every result per class and suggests a confidence threshold.

Released, the heads sent every one of the 600 check cases down the Overnight route (100 of 600, 17%);
trained, 448 of 600 (75%) took the right route and, where the route was right, every option was right.
Training gave 76% on its own held-back examples where a plain classifier on the same vectors
gave 89%: the heads have room to improve. The per-route counts, the questions and what is hardest:
[measurements](measurements.md#training-and-checking).

## The demo

```bash
uv run python examples/crowdstrike/replay_day.py examples/crowdstrike/scenario.toml --hub ATL
uv run python examples/crowdstrike/compare_models.py examples/crowdstrike/scenario.toml
```

| Script | What it does |
|---|---|
| `replay_day.py` | The coordinator: starts the server (with `--trace`), sends the day's messages through `decide_many` a simulated minute at a time, and redraws a scoreboard: decisions a second, how many were automatic and how many went to a person and why, holds made and lost, accuracy. It keeps its own copy of the seats and makes the holds the server reports, so each booking is judged against the seats left when it was decided. `--url` uses a server already running over HTTP. |
| `compare_models.py` | Gives CLM, a local language model and Claude the same 600 check cases: the message, the situation's facts, every rebooking route's journeys (fetched beforehand) and the policy's route descriptions. The language models answer with a route and a journey and call no tools. Claude runs at effort `low` with JSON output and the policy cached; a refusal counts as wrong. The Claude models need credentials (`ant auth login` or `ANTHROPIC_API_KEY`) and are skipped without them. |

### CLM against language models

On the same 600 check cases Claude Sonnet 5.5 took the right route for 494 (82%), Claude Haiku 5.5 for
473 (79%), CLM with the trained heads for 448 (75%) and Qwen3-4B, local, for 360 (60%). On this Mac,
Claude was both more accurate and faster (267 ms a decision against CLM's 300 ms); CLM's strengths are
that only 33 of its wrong routes were not marked uncertain, against Sonnet's 106, that the whole day
costs $0 against $158, and that the messages stay on this machine. The tables, the conditions and what
each model gets wrong: [measurements](measurements.md#clm-against-language-models).

### An agent's proposals, checked

```bash
uv run python examples/crowdstrike/check_calls.py examples/crowdstrike/scenario.toml
```

`check_calls.py` has `check_call` judge, from the message and the call alone, a rebooking and a refund
that another airline's agent proposes for the same passenger. As a yes or no, `check_call` is weak here:
114 of 200 caught, and no threshold does materially better; `decide` is the better check, right for 162
of the same 200. The table and the thresholds: [measurements](measurements.md#an-agents-proposals-checked).

### The day at Atlanta, replayed

Every booking starting at Atlanta, 9,704 bookings and 15,503 passengers, replayed on the Apple M5 Max:
3.6 decisions a second, 7,529 of 9,704 (78%) on the right route, 465 (4.8%) wrong and not marked
uncertain, 4,654 (48%) to a person, and 28 of 2,868 seat holds lost to a passenger decided just before.
Too many go to a person, and better-trained heads, not a lower threshold, are the answer. The table, the
per-route counts and the trace: [measurements](measurements.md#the-day-at-atlanta-replayed).

### The whole day on a data-centre GPU

The day's 99,806 decisions arrive over 41.5 hours, 10.7 a second at the busiest minute. With the trained
heads and every tool and action, the Apple M5 Max decided 3.6 a second, an NVIDIA A100 19.2 and an H100
51.1: one A100 keeps up with the whole day, the H100 decided all of it in 32.5 minutes, and the Mac
falls behind at the peaks. The H100 got slightly fewer right than the A100. The tables and each run's
detail: [measurements](measurements.md#the-whole-day-on-a-data-centre-gpu).
