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

On an Apple M5 Max (64 GB, MLX, the 8-bit CLM), trained on the 1,500 and checked on the 600 unseen
cases; each count is how many were right of all the cases with that answer:

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

33 routes (6%) were confidently wrong. Agent and Overnight are hardest: Agent covers two cases (too
vague, or a firm deadline nothing meets on a non-refundable fare), and Overnight needs several facts
together. Training gave 76% on its own held-back examples where a plain classifier on the same vectors
gave 89%: the heads have room to improve. Rail is not trained; only 180 of the day's 99,806 bookings
prefer it (north-east corridor only).

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

On an Apple M5 Max (64 GB): CLM and Qwen3-4B ran on MLX on that machine, Claude through the Anthropic
API with 8 calls at once; prices are those of 22 October 2026. Day-weighted accuracy weights each route
by its share of the day's right answers.

| | Claude Sonnet 5.5 | Claude Haiku 5.5 | CLM (trained heads) | Qwen3-4B (local) |
|---|---|---|---|---|
| Right route, of 600 | **494 (82%)** | 473 (79%) | 448 (75%) | 360 (60%) |
| Day-weighted accuracy | **84%** | 81% | 76% | 61% |
| Wrong and not marked uncertain | 106 | 127 | **33** | 240 |
| Unusable replies (counted as wrong) | 0 | 1 | 0 | 36 |
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

On this Mac, Claude was both more accurate and faster: about 2 seconds a call at 8 at once, and more
concurrency would be faster still, while CLM's time is almost all encoding on the Mac's GPU. On a
data-centre GPU CLM is much faster ([the whole day](#the-whole-day-on-a-data-centre-gpu), measured on
the replay, not here). CLM marks answers it is unsure of, so most of its wrong routes ran no action,
but many right answers go to a person too. The language models were not asked for a confidence (a
different harness, not measured), and not every wrong answer acts: an Agent answer hands over to a
person, an unusable reply runs nothing. Claude does worst on Refund and Agent, the routes that turn on what
the passenger wants or on a time nothing meets. The comparison cost about $1.05 in API calls (about
1,370 input tokens a case, 852 cached).

### An agent's proposals, checked

```bash
uv run python examples/crowdstrike/check_calls.py examples/crowdstrike/scenario.toml
```

Another airline's agent proposes both a rebooking (`hold_seats`) and a refund (`issue_refund`) for the
same passenger and asks `check_call` about each, with the message and the call only. The table covers
the 200 check cases whose right answer is Rebook or Refund, on the M5 Max with the trained heads, a call
approved at 0.5:

| Right answer | Rebooking approved | Refund approved | Caught: right approved, wrong refused | Right call has the higher probability |
|---|---|---|---|---|
| Rebook (100) | 62 | 29 | 36 | 82 |
| Refund (100) | 15 | 83 | 78 | 87 |

As a yes or no, `check_call` is weak here: 114 of 200 caught, and no threshold does materially better
(0.49 caught 116; 0.4, 111; 0.6, 103), because training covered the route and the questions, not
`check_call`'s own question. Compared with each other the proposals are told apart in 169 of 200.
`decide` is the better check: right for 162 of the same 200, with most wrong routes marked uncertain.

### The day at Atlanta, replayed

The replay took every booking starting at Atlanta, 9,704 bookings and 15,503 passengers, from 04:10 UTC
on 19 July to 07:36 UTC on 20 July, on the Apple M5 Max (MLX, trained heads):

| | |
|---|---|
| Time | 44 minutes in the server's `decide_many` calls, 3.6 decisions a second |
| Right route | 7,529 of 9,704 (78%) |
| Wrong and not marked uncertain | 465 (4.8%) |
| Automatic | 5,050 (52%) |
| To a person | 4,654 (48%): uncertain 4,142, Agent route 1,263, a hold lost 28 (a booking can have two reasons) |
| Seat holds | 2,868 made, 28 lost to a passenger decided just before |
| The coordinator's copy of the seats | took every hold the server made (checked one way only) |

Per route: Rebook 3,860 of 4,558, Rebook_Partner 73 of 94, Overnight 282 of 433, Refund 1,240 of 1,633,
Agent 812 of 1,454, No_Action 1,262 of 1,532.

The trace puts 100% of each `decide_many` call in CLM's encoding; the tools take about 3 ms a call. Too
many go to a person: the threshold of 0.6 keeps wrong-and-not-marked to 5% but marks 43% uncertain,
most of them right. Lowering it is not the answer, since `clm-evaluate --cases` suggests 0.63 as the
lowest threshold at which kept routes are 90% right; better-trained heads are. The 28 lost holds are
real races, failed visibly.

### The whole day on a data-centre GPU

The day's 99,806 decisions arrive over 41.5 hours; the table puts that demand next to what each machine
decided a second (trained heads, `decide_many` with every tool and action, as in the replay). The Mac's
rate is from Atlanta's replay, the A100's from the first 27,511 bookings (the run was stopped to change
machines), the H100's from the whole day. "Falls behind" is a minute-by-minute queue of arrivals at each
rate.

| | Decisions a second | Busiest minute's share of it | Falls behind |
|---|---|---|---|
| The day: average / busiest hour / busiest minute | 0.67 / 1.79 / 10.7 | | |
| Apple M5 Max (MLX, 8-bit) | 3.6 | 297% | in 84 minutes; at most 979 waiting, 4.5 minutes |
| NVIDIA A100 80 GB (torch, bf16) | 19.2 | 56% | never |
| NVIDIA H100 80 GB SXM (torch, bf16) | 51.1 | 21% | never |

One A100 keeps up with the whole day and an H100 decides 4.8 times what the busiest minute needs; the
Mac does not keep up at the peaks. The GPU is the limit: encoding is 96% of each call on both, the
server's one CPU core mostly waits. The H100 got slightly fewer right than the A100; training on a
different GPU or its rounding could each account for that, and neither was measured.

| | A100 | H100 |
|---|---|---|
| The run | first 27,511 bookings, 79% right | 32.5 minutes for all 99,806: 75,615 right (76%), 5,216 (5.2%) wrong and not marked uncertain, 46,030 (46%) uncertain (43% at Atlanta); 23,050 holds made, 123 lost. The later hours are harder than the first |
| First 3,000 bookings, rows of 1,024 / 2,048 / 4,096 tokens | 18.8 / 18.8 / 17.5 a second | 45.4 / 46.6 / 43.7 (2,048, the default, best on both) |
| Right on those 3,000, heads retrained on each machine | 2,399 | 2,379 (training 2 minutes 40 seconds, download included) |
| Before packing texts into rows without padding and ranking each request's options in one pass | 11.0 a second over the first 40,717 | |
| GPU use | | busy 82% of the run (sampled every 5 seconds), about 480 of 700 W, 16 of 80 GB |
| Two copies of the model on one GPU, same 3,000 | | 24.2 and 24.1 a second, 48.3 together against one copy's 46.6: one copy already fills it; more copies only make sense for different scenarios |
