# Writing a scenario

Your environment is the tools you define: queries that reach your data, actions, and your rules. A decision runs over
it in one of two styles: weighing options against your rules ([environment.md](environment.md)), or routing a message
with CLM. This file describes routing: `scenario.toml` says which routing decisions the server makes and which tools
they use. [examples/service_desk/scenario.toml](../examples/service_desk/scenario.toml) uses everything here. A mistake
in the file stops start-up with one line naming the setting.

## One decision

When an agent calls `decide` with the text of a request:

1. CLM picks the route from the text and, if `[routing] inform` names tools, their facts.
2. The route's informing tools run, and their facts are added to what CLM reads.
3. CLM answers the scenario's questions; the rules are checked against the answers.
4. The acting tools whose conditions hold run, unless the route is uncertain.
5. The reply goes back to the agent.

`decide_many` does this for many requests: CLM reads each step's texts in one batched pass. Every request's informing
tools run before any acting tool, so they see the state from before the batch's actions; the acting tools then run in
request order. When an acting tool changes what an informing tool reads, such as seats held for an earlier request, a
later request can be offered what is already taken, so its acting tool should fail cleanly (as `hold_seats` does).
Use `decide` when requests contend for the same thing.

## Routes

A route is one kind of request. You need at least two.

```toml
[routing]
instructions = "What does the requester need?"   # the question CLM is asked
confidence_threshold = 0.6                       # below this the decision is uncertain and no action runs

[routes.Data_Access]
description = "The requester needs access to an existing dataset or system."
examples = ["Can I get into the oncology registry?"]   # optional
inform = ["lookup_queue"]                              # optional: informing tools, in order
```

- **`description` is what CLM reads.** Its wording matters more than anything else in the file: use full sentences
  about the person's need, not labels. If you train CLM, changing a description, a route's name or the routing
  question means training again.
- **Include a route for everything else.** CLM always picks one, so without an "out of scope" route an unrelated
  request lands on a real one.
- **An uncertain decision is still returned**, with `uncertain: true`. Informing tools and questions run; acting tools
  do not. The agent can ask its user and call `decide` again with a clearer request. A route reused from the owner's
  correction is never uncertain.

## Questions for CLM

CLM can also answer other questions about each request.

```toml
[questions.identifiable]
type = "noul"                                    # yes or no; the answer is 0 to 1
instructions = "Does this request involve data that identifies individual patients?"

[questions.urgency]
type = "score"                                   # ordered levels; the answer is the expected level
instructions = "How soon is this needed?"
criteria = ["No deadline is mentioned", "Needed within weeks", "Needed within days"]

[questions.team]
type = "choice"                                  # one of several options
instructions = "Which team should handle this?"
criteria = { reporting = "Dashboards and reports", data = "Tables and data quality" }
```

The answers go to the agent and the acting tools. `clm-train` trains only questions your examples label
([fine-tuning.md](fine-tuning.md)); check the others by eye before relying on them.

## Rules

A rule is guidance that applies when an answer meets a condition. It goes to the agent, whose language model can
follow it, and to the acting tools.

```toml
[[rules]]
when = "urgency >= 1.5"
tell = "Set the ticket's priority to High and state the deadline in the summary."
```

`when` is `<question> >= <number>` or `<= <number>` for yes/no and score questions, and `<question> == <option>` for
choice questions.

## Tools

**Information from outside reaches a decision only through tools**, and a decision acts on the world only through
them. Both kinds are Python functions in a file beside the scenario.

```toml
[tools]
module = "service_desk_tools"      # service_desk_tools.py beside this file

[tools.settings]                   # passed to your module; any keys you like
tracker_url = "http://127.0.0.1:8765"
```

The module defines `TOOLS`, a list of functions, and optionally `configure(settings, scenario_dir)`, called once at
start-up.

Every tool takes one argument, the **request**, filled in as far as the decision has got:

```python
{"text": "...",                    # the caller's text
 "route": "Data_Access",
 "answers": {...},                 # CLM's answers to the questions
 "rules": [{"when": ..., "tell": ...}],
 "facts": ["...", ...],            # what the informing tools have found so far
 "context": {...}}                 # only if the caller sent one
```

A caller can send `context` with `decide`: whatever your tools need to find their records, such as a user, ticket or
booking ID. Tools get it as sent; CLM never reads it.

A tool returns anything JSON can hold. To report a failure, raise an exception with a message a person can act on: the
reply lists the tool as `failed` with that message, and the decision is still returned.

Give each tool a mock backend as well as the live one, switched by a setting such as `backend = "mock"`, so you can
train and check before connecting real systems. `clm-new-scenario` writes tools that way
([walkthrough.md](walkthrough.md#4-act-on-it-acting-tools)).

Make clients and connections once, in `configure()`: an HTTP client made on every call can cost more than CLM's part
of the decision. The server calls tools from several threads at once, so anything they share must allow that, as an
`httpx.Client` does.

### Informing tools

An informing tool returns `{"facts": ["a plain sentence", ...]}`, added to what CLM reads before answering the
questions. A route's own tools run after the route is chosen, so their facts never change it.

**Tools work things out; CLM judges.** CLM compares meanings; it does not calculate. The tool does the arithmetic and
lookups and states the result: "the table is tagged as personal data", "the connection will be missed", not raw
numbers.

Facts can sway unrelated questions. Each question hears only the facts of the tools that inform it:

```toml
[tools.inform.lookup_queue]
questions = []                     # given to no question; without this table, all of them
```

### Facts the route considers

When the right route depends on facts, such as whether a requested table holds personal data, name tools under
`[routing]`. They run first, and the route is decided with their facts:

```toml
[routing]
instructions = "What does the requester need?"
inform = ["table_tags"]            # run first; the route hears their facts
```

- The reply's `route`, `confidence` and `probabilities` are the decision made with the facts. `before_facts` holds the
  route from the message alone. Without `[routing] inform`, there is no `before_facts`.
- Their facts also go to the questions, as `[tools.inform]` says. A route that lists the same tool does not run it
  again.
- Corrections are saved, reused and trained on with the facts, the text the route saw. Send `correct` the same
  `context` as `decide`, so the tools find the same facts.
- `describe` lists these tools as `route_facts_from`.

### Options fetched at decision time

For a choice among things that exist only at that moment (which on-call engineer, runbook or knowledge-base article),
an informing tool finds the candidates and CLM chooses:

```toml
[routes.Page_On_Call]
description = "Someone needs to look at this now."
inform = ["on_call_now"]

[routes.Page_On_Call.options]
tool = "on_call_now"               # one of the route's informing tools
instructions = "Who is best placed to handle this?"
```

The tool returns `{"options": [{"id": "...", "text": "..."}, ...]}`, and may return `"facts"` too. The tool decides
what is possible, such as who is on call and free; CLM reads each option's text against the request and the facts. So
state what the tool worked out in plain words: "Sam, database team, on call until 18:00, two open pages".

- The reply gains `option`: `id`, `text`, `confidence`, `probabilities` and `uncertain` (below
  `[routing] confidence_threshold`), or `null` when the tool found none or returned something else. `informed` says
  which.
- The acting tools get the chosen option as `request["option"]` (`id` and `text`).
- On that route, actions are skipped with the reason `no option` or `option uncertain`.
- Routes without `options` have no `option` in their reply.
- The options tool runs after routing, so it cannot also be one of `[routing] inform`.

### Acting tools

An acting tool runs after the decision, on the routes you list, optionally only when a condition holds:

```toml
[[actions]]
tool = "create_ticket"
routes = ["Data_Access", "Research_Support", "Information_Request"]
when = "urgency >= 1.5"            # optional, in the rules' syntax
confirm = false                    # only false for now: the tool runs without asking
```

Every reply lists each action's status: `done` with the tool's result, `failed` with its message, or `skipped` with
the reason (`route uncertain`).

**Put every safeguard in the tool itself.** Validate input, check permissions and avoid duplicates there: the server
runs it whenever the decision says so, and a repeated call runs it again.

## Checking a caller's tool call

With `check_call`, an agent can ask CLM whether a tool call it proposes fits the request:

```toml
[tool_check]
instructions = "Is this tool call appropriate for what the user asked?"
threshold = 0.5          # at or above this the call is approved
```

CLM's opinion is advice. Untrained, it agreed with a reviewer only a little more than half the time.

## Corrections and reuse

```toml
[record]
live_corrections = true    # reuse a corrected route for a near-identical request
match_threshold = 0.97     # how similar "near-identical" must be, 0 to 1
```

In review mode the owner's corrections are saved to the record. Afterwards:

- a request close enough to a corrected one gets the owner's route, whatever CLM says;
- `correct` can also give the right answers (`"answers": {"urgency": 2, "identifiable": true}`) and, on a route with
  options, the right option's id. The route's informing tools run again, so send the same `context` as with `decide`;
- `clm-evaluate` replays the corrections, answers and options too, and suggests a confidence threshold;
- `clm-train` learns from them.

The record's location and encryption: [data-handling.md](data-handling.md). The models: [models.md](models.md).

## Other settings

| Setting | Default | Meaning |
|---|---|---|
| `[clm] backend` | `"mlx"` | `"mlx"` on a Mac, `"torch"` on Linux with a GPU: see [models.md](models.md) |
| `[clm] model_id` | the 8-bit MLX build, or the authors' release with torch | Which CLM to download |
| `[clm] device` | the first of cuda, mps, cpu | Torch only: where CLM runs |
| `[clm] heads` | none | Folder of trained heads, beside the scenario |
| `[generator]` | MLX Qwen3-4B | The language model for `clm-synthesize`, not the server. On Linux, use `ollama` or `openai` |
| `[synthesize] messages` | "messages that a person might send to an assistant" | What `clm-synthesize` asks for: say who is writing and to what |
