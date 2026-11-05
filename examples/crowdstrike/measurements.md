# The CrowdStrike day: measurements

This page holds the measured results for [the CrowdStrike example](README.md): the trained heads checked, CLM
against language models, an agent's proposals checked, and the day replayed on a Mac and on data-centre GPUs,
with their conditions.

## Training and checking

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
together. Rail is not trained; only 180 of the day's 99,806 bookings prefer it (north-east corridor only).

## CLM against language models

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

## An agent's proposals, checked

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

## The day at Atlanta, replayed

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

## The whole day on a data-centre GPU

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
