# Service desk example

Decisions for a data and analytics service desk. Someone working in a clinical environment
describes what they need; the decision server works out what kind of request it is, how urgent
it is and whether it involves identifiable data, looks up which team handles it, and opens a
ticket in a tracker.

It is the small finished scenario to run first, and then to copy for your own decisions. It is
not a clinical tool. All data here is invented.

## Run it

First train CLM for the scenario, once; the repository does not include trained heads (see
[Training CLM for the scenario](#training-clm-for-the-scenario)):

```bash
uv run python train.py examples/service_desk/scenario.toml
```

Then two terminals, from the project root.

```bash
uv run python examples/service_desk/mock_tracker.py                    # terminal 1: the mock tracker
uv run python chat.py examples/service_desk/scenario.toml --review     # terminal 2: the test chat
```

Try:

- `I need access to the oncology registry for an audit, needed this week`: a ticket with
  `priority-high` and `governance-review`.
- `Should this patient be started on a beta blocker?`: out of scope, so no ticket.
- `can you help with the audit data`: a vaguer request, which CLM is less sure of. If its route
  is wrong, type `/correct Data_Access`; send it again and your route is used.
- `/agreement`, then `/routes`, then `/quit`.

**Every handled request opens a ticket, at once.** Typing the same request twice opens two.

## What decides what

| Decision | Made by |
|---|---|
| The kind of request | CLM; you can correct it in review mode, and the correction is reused |
| Who handles it, what approvals it needs, how long it takes | `lookup_queue`, from `queues.json` |
| How urgent it is, and whether identifiable data is involved | CLM |
| Whether to open a ticket | The `[[actions]]` entry: every route except Out_Of_Scope, unless CLM is unsure of the route |
| The ticket's priority and labels | `create_ticket`, from CLM's answers |
| What to tell the requester | The calling agent; the server returns the rules for it to follow |

## The files

| File | What it is |
|---|---|
| `scenario.toml` | The routes, CLM's questions, the rules, the tools and their settings |
| `service_desk_tools.py` | The tools: `lookup_queue` informs, `create_ticket` acts |
| `queues.json` | Which team handles each kind of request |
| `scenario.examples.jsonl` | Labelled example messages for training |
| `mock_tracker.py` | A stand-in for an issue tracker's REST interface (Forgejo's) |

## Training CLM for the scenario

As released, CLM routes poorly, and the repository does not include trained heads: they are
made on the machine that serves them. One command fits CLM to this scenario in under a minute:

```bash
uv run python train.py examples/service_desk/scenario.toml
```

That learns from `scenario.examples.jsonl` (example messages written by a language model and
shipped here; those for routes the scenario no longer has are left out) and from any routes
you have corrected. It saves the result in `scenario.heads/`, which the server picks up at its
next start.

To write a fresh set of examples, delete `scenario.examples.jsonl` and run
`uv run python synthesize.py examples/service_desk/scenario.toml`.

What it did here: on 55 held-back generated examples, 51% with the released heads and 98%
trained. Through the server, the trained heads routed 15 of 17 hand-written test messages
correctly (`tests/desk_routing_cases.py`). The generated examples are easier than real
messages, and 17 messages is a small test.

**Change a route's description, a route's name or the routing question and you must train
again.** The server notices, says so at start-up, and uses the released heads until you do.

## Using a real tracker instead of the mock

The tracker tools speak the REST API of [Forgejo](https://forgejo.org) (Gitea has the same
one), and the mock imitates it. Forgejo is a single program with a built-in database. This
has been run against Forgejo 16.

1. Install and start it: `brew install forgejo`, then `brew services start forgejo`.
2. Open http://localhost:3000 and press "Install Forgejo" (the defaults are fine), creating an
   administrator account at the bottom of that page.
3. Create a repository called `requests`. Its issues are the tickets.
4. In your user settings, under Applications, create an access token with read and write
   access to issues.
5. In `[tools.settings]` set `tracker_url = "http://localhost:3000"` and
   `repository = "<your user>/requests"`.
6. In the terminal where you start the chat or the server: `export TRACKER_TOKEN=<the token>`.

The server then creates real issues, which you can see in the browser. Priority and the
governance flag become labels (`priority-high`, `governance-review`); a label the repository
does not have is created.

For a different tracker, rewrite `create_ticket` and its helpers in `service_desk_tools.py`.
Nothing else depends on them.

## Limitations

- **Routing is only reliable after training**, and even then expect some misses.
- **Nothing stops someone typing patient details**, and `create_ticket` copies the request's
  text into the ticket. The corrections record beside the scenario stores corrected text too.
  See [data-handling.md](../../docs/data-handling.md).
- **No duplicate check.** The same request sent twice opens two tickets.
