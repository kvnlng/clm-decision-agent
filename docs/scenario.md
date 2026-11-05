# Writing a scenario

`scenario.toml` says which routing decisions the server makes and which tools they use. Weighing options against
rules instead is [environment.md](environment.md). [examples/service_desk/scenario.toml](../examples/service_desk/scenario.toml) uses
everything here; [walkthrough.md](walkthrough.md) builds one from nothing. A mistake in the file stops start-up with
one line naming the setting.

## One decision

When an agent calls `decide` with the text of a request:

1. CLM picks the route from the text and, if `[routing] inform` names tools, their facts.
2. The route's informing tools run; their facts are added to what CLM reads.
3. CLM answers the questions; the rules are checked against the answers.
4. The acting tools whose conditions hold run, unless the route is uncertain.
5. The reply goes back to the agent.

`decide_many` does this for many requests, CLM reading each step's texts in one batched pass. Every request's
informing tools run before any acting tool, then the acting tools run in request order. So when an action changes
what a tool reads (seats held for an earlier request), a later request can be offered what is already taken, and its
acting tool should fail cleanly, as `hold_seats` does. Use `decide` when requests contend for the same thing.

## Routes

A route is one kind of request. You need at least two, one of them for everything else: CLM always picks a route,
so without an "out of scope" route an unrelated request lands on a real one.

```toml
[routing]
instructions = "What does the requester need?"
confidence_threshold = 0.6
inform = ["table_tags"]

[routes.Data_Access]
description = "The requester needs access to an existing dataset or system."
examples = ["Can I get into the oncology registry?"]
inform = ["lookup_queue"]
```

| Key | Type | Default | Meaning |
|---|---|---|---|
| `[routing] instructions` | string | required | The question CLM is asked |
| `[routing] confidence_threshold` | 0 to 1 | `0.6` | Below this the decision is uncertain and no action runs |
| `[routing] inform` | tool names | `[]` | Run before routing; the route hears their facts ([below](#facts-the-route-considers)) |
| `[routes.<name>] description` | string | required | What CLM reads: full sentences about the person's need, not labels. Its wording matters more than anything else in the file |
| `[routes.<name>] examples` | strings | `[]` | Example messages |
| `[routes.<name>] inform` | tool names | `[]` | The route's informing tools, in order |
| `[routes.<name>.options]` | table | none | A choice fetched at decision time ([below](#options-fetched-at-decision-time)) |

An uncertain decision is still returned, with `uncertain: true`: informing tools and questions run, acting tools do
not, and the agent can ask its user and call `decide` again. If you train CLM, changing a description, a route's name
or the routing question means training again.

## Questions for CLM

CLM can answer other questions about each request. The answers go to the agent and the acting tools.

```toml
[questions.identifiable]
type = "noul"
instructions = "Does this request involve data that identifies individual patients?"

[questions.urgency]
type = "score"
instructions = "How soon is this needed?"
criteria = ["No deadline is mentioned", "Needed within weeks", "Needed within days"]

[questions.team]
type = "choice"
instructions = "Which team should handle this?"
criteria = { reporting = "Dashboards and reports", data = "Tables and data quality" }
```

| Key | Meaning |
|---|---|
| `type` | `"noul"`: yes or no, answered 0 to 1. `"score"`: ordered levels, answered with the expected level. `"choice"`: one of several options |
| `instructions` | The question CLM is asked |
| `criteria` | `score`: at least two levels, lowest first. `choice`: a table of at least two options with descriptions. Not used by `noul` |

`clm-train` trains only questions your examples label ([fine-tuning.md](fine-tuning.md)); check the others by eye
before relying on them.

## Rules

A rule is guidance that applies when an answer meets a condition. It goes to the agent, whose language model can
follow it, and to the acting tools.

```toml
[[rules]]
when = "urgency >= 1.5"
tell = "Set the ticket's priority to High and state the deadline in the summary."
```

| Key | Meaning |
|---|---|
| `when` | `<question> >= <number>` or `<= <number>` for `noul` and `score` questions; `<question> == <option>` for `choice` questions |
| `tell` | The guidance |

## Tools

**Information from outside reaches a decision only through tools**, and a decision acts on the world only through
them. Both kinds are Python functions in a module beside the scenario.

```toml
[tools]
module = "service_desk_tools"      # service_desk_tools.py beside this file

[tools.settings]                   # passed to your module; any keys you like
tracker_url = "http://127.0.0.1:8765"

[tools.inform.lookup_queue]
questions = []                     # given to no question; without this table, all of them
```

| Key | Default | Meaning |
|---|---|---|
| `[tools] module` | none | Defines `TOOLS`, a list of functions, and optionally `configure(settings, scenario_dir)`, called once at start-up. Loaded from the file beside the scenario, so no other module of that name can stand in for it; a name with no file there is imported as an installed package |
| `[tools.settings]` | `{}` | Passed to `configure()` |
| `[tools.inform.<tool>] questions` | all questions | Which questions hear the tool's facts: facts can sway unrelated questions |

Every tool takes one argument, the **request**, filled in as far as the decision has got:

```python
{"text": "...",                    # the caller's text
 "route": "Data_Access",
 "answers": {...},                 # CLM's answers to the questions
 "rules": [{"when": ..., "tell": ...}],
 "facts": ["...", ...],            # what the informing tools have found so far
 "context": {...}}                 # only if the caller sent one: what your tools need to find their
                                   # records (a user, ticket or booking ID), as sent; CLM never reads it
```

A tool returns anything JSON can hold. To report a failure, raise an exception with a message a person can act on:
the reply lists the tool as `failed` with it, and the decision is still returned.

Give each tool a mock backend as well as the live one, switched by a setting such as `backend = "mock"`, to train and
check before connecting real systems; `clm-new-scenario` writes tools that way
([walkthrough.md](walkthrough.md#4-act-on-it-acting-tools)). Make clients and connections once, in `configure()`: an
HTTP client made on every call can cost more than CLM's part of the decision. Tools are called from several threads
at once, so anything they share must allow that, as an `httpx.Client` does.

### Informing tools

An informing tool returns `{"facts": ["a plain sentence", ...]}`, added to what CLM reads before answering the
questions. A route's own tools run after the route is chosen, so their facts never change it.

**Tools work things out; CLM judges.** CLM compares meanings; it does not calculate. The tool does the arithmetic and
lookups and states the result: "the table is tagged as personal data", "the connection will be missed", not raw
numbers.

### Facts the route considers

When the right route depends on facts, such as whether a requested table holds personal data, name tools under
`[routing] inform`. They run first, and the route is decided with their facts.

- The reply's `route`, `confidence` and `probabilities` are the decision made with the facts; `before_facts` holds
  the route from the message alone, and is absent without `[routing] inform`.
- Their facts also go to the questions, as `[tools.inform]` says. A route that lists the same tool does not run it
  again.
- Corrections are saved, reused and trained on with the facts, the text the route saw.
- `describe` lists these tools as `route_facts_from`.

### Options fetched at decision time

For a choice among things that exist only at that moment (which on-call engineer, runbook or knowledge-base
article), an informing tool finds the candidates and CLM chooses:

```toml
[routes.Page_On_Call]
description = "Someone needs to look at this now."
inform = ["on_call_now"]

[routes.Page_On_Call.options]
tool = "on_call_now"
instructions = "Who is best placed to handle this?"
```

| Key | Meaning |
|---|---|
| `tool` | One of the route's informing tools. It runs after routing, so it cannot also be in `[routing] inform` |
| `instructions` | The question CLM answers over the options |

The tool returns `{"options": [{"id": "...", "text": "..."}, ...]}`, and may return `"facts"` too. It decides what
is possible, such as who is on call and free; CLM reads each option's text against the request and the facts, so
the text states what the tool worked out: "Sam, database team, on call until 18:00, two open pages".

- The reply gains `option`: `id`, `text`, `confidence`, `probabilities` and `uncertain` (below
  `[routing] confidence_threshold`), or `null` when the tool found none or returned something else; `informed` says
  which. Routes without `options` have no `option` in their reply.
- The acting tools get the chosen option as `request["option"]` (`id` and `text`), or are skipped with the reason
  `no option` or `option uncertain`.

### Acting tools

An acting tool runs after the decision, on the routes you list, optionally only when a condition holds:

```toml
[[actions]]
tool = "create_ticket"
routes = ["Data_Access", "Research_Support", "Information_Request"]
when = "urgency >= 1.5"
confirm = false
```

| Key | Default | Meaning |
|---|---|---|
| `tool` | required | The function to run |
| `routes` | required, at least one | The routes it runs on |
| `when` | none | In the rules' syntax; the tool runs only when it holds |
| `confirm` | `false` | Only `false` for now: the tool runs without asking |

Every reply lists each action's status: `done` with the tool's result, `failed` with its message, or `skipped` with
the reason (`route uncertain`, `no option`, `option uncertain`, `read-only server`).

**Put every safeguard in the tool itself.** Validate input, check permissions and avoid duplicates there: the server
runs it whenever the decision says so, and a repeated call runs it again.

## Checking a caller's tool call

With `check_call`, an agent sends a request's text and a tool call it proposes, and gets CLM's probability that the
call fits and whether it is approved.

| Key | Default | Meaning |
|---|---|---|
| `[tool_check] instructions` | `"Is this tool call appropriate for what the user asked?"` | The question CLM is asked |
| `[tool_check] threshold` | `0.5` | At or above this (0 to 1) the call is approved |

CLM's opinion is advice. Training covers the routes and questions, not this question. On the CrowdStrike example
(an M5 Max, trained heads, approved at 0.5, the 200 check cases whose answer is Rebook or Refund, a rebooking and a
refund proposed for each) it approved the right call and refused the wrong one for 114 of 200
([examples/crowdstrike](../examples/crowdstrike/README.md), [findings.md](findings.md)).

## Corrections and reuse

With `clm-server --review`, the server also offers `correct` and `agreement`. `correct` saves the owner's route for
a request to the record, optionally with the right answers (`"answers": {"urgency": 2, "identifiable": true}`) and,
on a route with options, the right option's id; the route's informing tools run again, so send the same `context` as
with `decide`, and the tools find the same facts. `agreement` reports how often CLM's own route agrees with the saved corrections. A route reused from a
correction is never uncertain.

| Key | Default | Meaning |
|---|---|---|
| `[record] live_corrections` | `true` | A request close enough to a corrected one gets the owner's route, whatever CLM says |
| `[record] match_threshold` | `0.97` | How similar "close enough" must be, 0 to 1 |
| `[record] path` | `<scenario>.corrections.jsonl` | Where the record is written, relative to the scenario |
| `[record] key_env` or `key_file` | none | The record's encryption key, in a variable or a file beside the scenario: [data-handling.md](data-handling.md) |

`clm-evaluate` replays the corrections, answers and options and suggests a confidence threshold; `clm-train` learns
from them ([fine-tuning.md](fine-tuning.md)).

## Other settings

| Key | Default | Meaning |
|---|---|---|
| `[clm] backend` | `"mlx"` | `"mlx"` on a Mac, `"torch"` on Linux with a GPU: [models.md](models.md) |
| `[clm] model_id` | the 8-bit MLX build, or the authors' release with torch | Which CLM to download |
| `[clm] device` | the first of `cuda`, `mps`, `cpu` | Torch only: where CLM runs |
| `[clm] heads` | none | Folder of trained heads, relative to the scenario |
| `[generator] backend` | `"mlx"` | `"mlx"`, `"ollama"` or `"openai"`: the language model for `clm-synthesize`, not the server. On Linux, use `ollama` or `openai` |
| `[generator] model` | MLX Qwen3-4B | Its model |
| `[generator] max_tokens` | `400` | Its reply length |
| `[generator] base_url`, `api_key_env` | none | With `openai`: the server's http(s) address, and the variable holding its key (never a key in the URL) |
| `[synthesize] messages` | `"messages that a person might send to an assistant"` | What `clm-synthesize` asks for: say who is writing and to what |
