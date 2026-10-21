# Writing a scenario

A scenario is one TOML file that says which decisions the server makes. Start from
[examples/service_desk/scenario.toml](../examples/service_desk/scenario.toml), which uses
everything described here. A mistake in the file stops start-up with one line naming the
setting.

## One decision

When an agent calls `decide` with the text of a request:

1. CLM picks the route from the text alone.
2. The route's informing tools run, and their facts are added to what CLM reads.
3. CLM answers the scenario's questions; the rules are checked against the answers.
4. The acting tools whose conditions hold run, unless the route is uncertain.
5. The reply goes back to the agent.

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

- **`description` is what CLM reads.** It compares the request with each description, so
  wording matters more than anything else in the file. Full sentences that describe the
  person's need work better than labels. If you train CLM, changing a description, a route's
  name or the routing question means training again.
- **Include a route for everything else.** CLM always picks one of the routes, so without an
  "out of scope" route an unrelated request lands on a real one.
- **An uncertain decision is still returned**, with `uncertain: true`. Its informing tools and
  questions run, so the agent sees everything, but no acting tool runs. The agent can ask its
  own user and call `decide` again with a clearer request. A route reused from the owner's
  correction is never uncertain.

## Questions for CLM

Besides the route, CLM can answer other questions about each request.

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

The answers are returned to the agent and given to the acting tools. They are not trained by
`train.py`; check them by eye before relying on them.

## Rules

A rule is guidance that applies when one of CLM's answers meets a condition. It is returned to
the agent, whose language model can follow it, and given to the acting tools.

```toml
[[rules]]
when = "urgency >= 1.5"
tell = "Set the ticket's priority to High and state the deadline in the summary."
```

`when` is `<question> >= <number>` or `<= <number>` for yes/no and score questions, and
`<question> == <option>` for choice questions.

## Tools

**Information from outside reaches a decision only through tools**, and a decision acts on the
world only through tools. Both kinds are ordinary Python functions in a file beside the
scenario.

```toml
[tools]
module = "service_desk_tools"      # service_desk_tools.py beside this file

[tools.settings]                   # passed to your module; any keys you like
tracker_url = "http://127.0.0.1:8765"
```

The module defines `TOOLS`, a list of functions, and optionally
`configure(settings, scenario_dir)`, which is called once at start-up.

Every tool takes one argument, the **request**, filled in as far as the decision has got:

```python
{"text": "...",                    # the caller's text
 "route": "Data_Access",
 "answers": {...},                 # CLM's answers to the questions
 "rules": [{"when": ..., "tell": ...}],
 "facts": ["...", ...]}            # what the informing tools have found so far
```

It returns anything JSON can hold. To report a failure, raise an exception with a message a
person can act on: the reply lists that tool as `failed` with the message, and the decision is
still returned.

### Informing tools

An informing tool returns `{"facts": ["a plain sentence", ...]}`. Facts are added to what CLM
reads before it answers the questions; they never reach the routing decision, so trained heads
and corrections are unaffected.

Facts can sway questions they have nothing to do with, so say which questions they bear on.
Each question hears only the facts of the tools that inform it:

```toml
[tools.inform.lookup_queue]
questions = []                     # given to no question; without this table, all of them
```

### Acting tools

An acting tool runs after the decision, on the routes you list and, optionally, only when a
condition holds:

```toml
[[actions]]
tool = "create_ticket"
routes = ["Data_Access", "Research_Support", "Information_Request"]
when = "urgency >= 1.5"            # optional, in the rules' syntax
confirm = false                    # only false for now: the tool runs without asking
```

Every reply lists the actions with their status: `done` with the tool's result, `failed` with
its message, or `skipped` with the reason (`route uncertain`).

**Put every safeguard in the tool itself.** Validate what it is given, check permissions and
avoid duplicates there: the server runs it whenever the decision says so, and a repeated call
runs it again.

## Checking a caller's tool call

An agent can ask CLM whether a tool call it proposes fits the request, with `check_call`:

```toml
[tool_check]
instructions = "Is this tool call appropriate for what the user asked?"
threshold = 0.5          # at or above this the call is approved
```

CLM's opinion is advice. Untrained, it agreed with a reviewer only a little more than half the
time.

## Corrections and reuse

```toml
[record]
live_corrections = true    # reuse a corrected route for a near-identical request
match_threshold = 0.97     # how similar "near-identical" must be, 0 to 1
```

In review mode the owner's corrections are saved to the record. Afterwards:

- a request close enough to a corrected one gets the owner's route, whatever CLM says;
- `evaluate.py` replays the corrections and suggests a confidence threshold;
- `train.py` learns from them.

Where the record is kept and how to encrypt it: [data-handling.md](data-handling.md). The
models: [models.md](models.md).

## Other settings

| Setting | Default | Meaning |
|---|---|---|
| `[clm] backend` | `"mlx"` | `"mlx"` on a Mac, `"torch"` on Linux with a GPU: see [models.md](models.md) |
| `[clm] model_id` | the 8-bit MLX build, or the authors' release with torch | Which CLM to download |
| `[clm] device` | the first of cuda, mps, cpu | Torch only: where CLM runs |
| `[clm] heads` | none | Folder of trained heads, beside the scenario |
| `[generator]` | MLX Qwen3-4B | The language model `synthesize.py` uses; not used by the server. On Linux, use `ollama` or `openai` |
| `[synthesize] messages` | "messages that a person might send to an assistant" | What `synthesize.py` asks the language model to write: say who is writing and to what |
