# Writing a scenario

`scenario.toml` says which routing decisions the server makes and which tools they use. Weighing options against rules
instead is [environment.md](environment.md).
[examples/service_desk/scenario.toml](../examples/service_desk/scenario.toml) uses everything here;
[walkthrough.md](walkthrough.md) builds one from nothing. A mistake in the file stops start-up with one line naming the
setting. The key tables and the exact rules are in [scenario-reference.md](scenario-reference.md).

## One decision

When an agent calls `decide` with the text of a request:

1. CLM picks the route from the text and, if `[routing] inform` names tools, their facts.
2. The route's informing tools run; their facts are added to what CLM reads.
3. CLM answers the questions; the rules are checked against the answers.
4. The acting tools whose conditions hold run, unless the route is uncertain.
5. The reply goes back to the agent.

<p align="center"><img src="images/routing-decision.svg" alt="One routing decision: the agent calls decide; CLM picks the route; the informing tools add facts and an options tool fetches candidates; CLM answers the questions, the rules are checked and an option is chosen. If the route and option are certain and the server is not read-only, the acting tools whose conditions hold run. The reply goes back to the agent." width="760"></p>

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

An uncertain decision is still returned, with `uncertain: true`: informing tools and questions run, acting tools do
not, and the agent can ask its user and call `decide` again. If you train CLM, changing a description, a route's name
or the routing question means training again. The keys: [reference](scenario-reference.md#routes).

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

The keys and what training covers: [reference](scenario-reference.md#questions-for-clm).

## Rules

A rule is guidance that applies when an answer meets a condition. It goes to the agent, whose language model can
follow it, and to the acting tools.

```toml
[[rules]]
when = "urgency >= 1.5"
tell = "Set the ticket's priority to High and state the deadline in the summary."
```

The `when` syntax: [reference](scenario-reference.md#rules).

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

The keys, how the module is loaded, and mock backends, clients and threads: [reference](scenario-reference.md#tools).

### Informing tools

An informing tool returns `{"facts": ["a plain sentence", ...]}`, added to what CLM reads before answering the
questions. A route's own tools run after the route is chosen, so their facts never change it.

**Tools work things out; CLM judges.** CLM compares meanings; it does not calculate. The tool does the arithmetic and
lookups and states the result: "the table is tagged as personal data", "the connection will be missed", not raw
numbers.

### Facts the route considers

When the right route depends on facts, such as whether a requested table holds personal data, name tools under
`[routing] inform`. They run first, and the route is decided with their facts. What the reply holds and where the
facts go: [reference](scenario-reference.md#facts-the-route-considers).

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

The keys, what the tool returns and what the reply gains:
[reference](scenario-reference.md#options-fetched-at-decision-time).

### Acting tools

An acting tool runs after the decision, on the routes you list, optionally only when a condition holds:

```toml
[[actions]]
tool = "create_ticket"
routes = ["Data_Access", "Research_Support", "Information_Request"]
when = "urgency >= 1.5"
confirm = false
```

**Put every safeguard in the tool itself.** Validate input, check permissions and avoid duplicates there: the server
runs it whenever the decision says so, and a repeated call runs it again.

The keys and the action statuses: [reference](scenario-reference.md#acting-tools).

## Checking a caller's tool call

With `check_call`, an agent sends a request's text and a tool call it proposes, and gets CLM's probability that the
call fits and whether it is approved. The keys and how well it did:
[reference](scenario-reference.md#checking-a-callers-tool-call).

## Corrections and reuse

With `clm-server --review`, the server also offers `correct` and `agreement`. What each does, the `[record]` keys
and what `clm-evaluate` and `clm-train` do with the record: [reference](scenario-reference.md#corrections-and-reuse).

## Other settings

The `[clm]`, `[generator]` and `[synthesize]` keys: [reference](scenario-reference.md#other-settings).
