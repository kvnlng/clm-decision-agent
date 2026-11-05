# Service desk example

A data and analytics service desk: someone in a clinical environment describes what they need; the server works
out the kind of request, its urgency and whether it involves identifiable data, looks up the team and opens a
ticket. The small scenario to run first, then copy. Not a clinical tool; all data invented.

## Run it

Train CLM once ([Training CLM for the scenario](#training-clm-for-the-scenario)), then two terminals from the
project root:

```bash
uv run clm-train examples/service_desk/scenario.toml

uv run python examples/service_desk/mock_tracker.py                     # terminal 1: the mock tracker
npx -y @modelcontextprotocol/inspector@2.13.0 \
    uv run clm-server examples/service_desk/scenario.toml --review        # terminal 2: the MCP Inspector
```

In the Inspector ([getting started](../../docs/getting-started.md#looking-inside-the-server)), connect and call
`decide` with each `text`:

- `I need access to the oncology registry for an audit, needed this week`: a ticket, `priority-high`,
  `governance-review`.
- `Should this patient be started on a beta blocker?`: out of scope, no ticket.
- `can you help with the audit data`: vaguer, CLM less sure. If wrong, `correct` with the same text and `route`
  Data_Access; `decide` again uses it.
- `agreement`, then `describe` for the routes and questions.

**Every handled request opens a ticket, at once**; the same request twice opens two.

## What decides what

| Decision | Made by |
|---|---|
| Kind of request | CLM (correctable in review mode; corrections reused) |
| Team, approvals, how long | `lookup_queue`, from `queues.json` |
| Urgency; identifiable data | CLM |
| Open a ticket? | `[[actions]]`: every route but Out_Of_Scope, unless CLM is unsure |
| Ticket priority and labels | `create_ticket`, from CLM's answers |
| What to tell the requester | The calling agent, from the server's rules |

## The files

| File | What it is |
|---|---|
| `scenario.toml` | Routes, questions, rules, tools, settings |
| `service_desk_tools.py` | `lookup_queue` informs, `create_ticket` acts |
| `queues.json` | Which team handles each kind of request |
| `scenario.examples.jsonl` | Labelled training messages |
| `mock_tracker.py` | Mock of Forgejo's issue REST interface |

## Training CLM for the scenario

As released, CLM routes poorly. `clm-train` (under [Run it](#run-it)) fits heads where they are served, in under
a minute, from `scenario.examples.jsonl` (routes the scenario no longer has are left out) and your corrections,
and saves `scenario.heads/` for the server's next start. For fresh examples, delete
`scenario.examples.jsonl` and run `uv run clm-synthesize examples/service_desk/scenario.toml`.

| Measure | Released heads | Trained heads |
|---|---|---|
| 55 held-back generated examples | 51% | 98% |
| 17 hand-written messages through the server (`tests/desk_routing_cases.py`) | | 15 of 17 |

Generated examples are easier than real messages, and 17 is a small test. **Change a route's description or
name, or the routing question, and train again**; the server says so at start-up and uses the released heads
meanwhile.

## Using a real tracker instead of the mock

The tools speak [Forgejo](https://forgejo.org)'s REST API (also Gitea's). Tested with Forgejo 16:

1. `brew install forgejo`, then `brew services start forgejo`.
2. Open http://localhost:3000, press "Install Forgejo", create an administrator account.
3. Create a repository `requests`: its issues are the tickets.
4. User settings, Applications: an access token with read and write on issues.
5. `[tools.settings]`: `tracker_url = "http://localhost:3000"`, `repository = "<your user>/requests"`.
6. Then `export TRACKER_TOKEN=<the token>` before starting the chat or server.

Priority and the governance flag become labels (`priority-high`, `governance-review`), created if missing. For
another tracker, rewrite `create_ticket` and its helpers in `service_desk_tools.py`; nothing else uses them.

## Limitations

- **Routing is only reliable after training**, and even then expect misses.
- **Nothing stops someone typing patient details**; `create_ticket` copies the text into the ticket, and the
  corrections record stores corrected text ([data-handling.md](../../docs/data-handling.md)).
- **No duplicate check.** The same request twice opens two tickets.
