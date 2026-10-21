# CLM decision server

A small framework for building a **decision server**: a program that makes narrow, checkable
decisions with a Contrastive Language Model (CLM) and offers them to other agents over
[MCP](https://modelcontextprotocol.io). You describe the decisions in one file, train CLM on
them, and run the server. An agent, for example one your colleagues build on Databricks, then
calls it whenever it needs that decision made.

It is a prototype, built to learn how CLM behaves and to show the pattern to people who want
to adapt it. It has not been used in production.

**It is not for clinical decisions.** The example is a data service desk in a clinical
setting; it routes questions about patient care out of scope.

## The idea

A language model generates text. A CLM does not: it reads a situation and a list of options
and says which option fits, with a probability. That makes its decisions cheap, fast, and easy
to log, test and train. The calling agent keeps doing what a language model is good at, such as
talking and writing, and leans on the decision server for the choices.

| Job | Done by |
|---|---|
| Which kind of request is this? How urgent? Does it involve identifiable data? Is this tool call appropriate? | CLM, in the decision server |
| Facts that bear on a decision, such as who handles the request | The scenario's informing tools |
| Acting on a decision, such as opening a ticket | The scenario's acting tools |
| Talking to the person, asking follow-up questions, writing replies | The calling agent |
| Was CLM's decision right? | The owner, whose corrections are saved and trained on |

All of it is driven by one file, the **scenario**: the kinds of request (routes), the questions
CLM answers, the rules that follow from the answers, and the tools. Information from outside
reaches a decision only through those tools. There is no language model in the server.

```
test chat  ──MCP──▶  decision server ──▶ CLM (in this process: MLX on a Mac, PyTorch on a GPU)
(or any agent)            ├──▶ informing tools (facts)
                          └──▶ acting tools (do things)
```

## Getting started

You need Python 3.12 or later, [uv](https://docs.astral.sh/uv/), and one of:

- **a Mac with Apple Silicon** and 16 GB of memory or more. CLM runs with MLX; the first run
  downloads about 8 GB.
- **Linux with an NVIDIA GPU of 24 GB or more**, such as an A100. CLM runs with PyTorch; the
  first run downloads about 16 GB. Before step 2, switch the example to the torch backend: in
  `examples/service_desk/scenario.toml`, comment out the MLX lines under `[clm]` and uncomment
  the torch ones. A plain CPU works too, slowly. There is a `Dockerfile`:
  [docs/serving.md](docs/serving.md#in-docker).

**1. Install.** The same command on both machines.

```bash
uv sync
```

**2. Run the example.** It is a small finished scenario: a data service desk.

```bash
uv run python examples/service_desk/mock_tracker.py                          # terminal 1: a stand-in issue tracker
uv run python chat.py examples/service_desk/scenario.toml --review           # terminal 2: the test chat
```

The test chat starts the decision server and talks to it over MCP, as an outside agent would.
Type a request and it shows the decision:

```
you> I need access to the oncology registry for an audit, needed this week
  route: Data_Access 0.65 (next: Information_Request 0.23)
  answers: urgency 1.61   identifiable 0.54
  facts: lookup_queue: Data Governance handles this kind of request. ...
  rules: urgency >= 1.5: Set the ticket's priority to High and state the deadline in the summary.
  rules: identifiable >= 0.5: Add the label governance-review and say that data governance ...
  actions: create_ticket done: {"number": 5, "url": "http://127.0.0.1:8765/..."}
```

`/routes` shows the scenario's routes and questions, and `/quit` ends the chat.
[examples/service_desk/README.md](examples/service_desk/README.md) walks through it.

**3. Write your own scenario.** Copy `examples/service_desk/` to a folder of your own and edit
`scenario.toml` and the tools file beside it: [docs/scenario.md](docs/scenario.md) explains
every part.

**4. Check it.** Start the chat with your scenario. A mistake in the file stops start-up with
one line naming the setting.

**5. Correct decisions.** In review mode (`--review`), type `/correct <route>` after a wrong
decision. The correction is saved, and a near-identical request gets your route straight away.
`/agreement` shows how often CLM's own route agrees with your corrections.

**6. Train.** Optionally have a language model write example messages for each route, then
train CLM on them and on your corrections, and see agreement rise:

```bash
uv run python synthesize.py my_scenario.toml      # optional; needs a language model
uv run python train.py my_scenario.toml
uv run python evaluate.py my_scenario.toml
```

[docs/fine-tuning.md](docs/fine-tuning.md) has the details.

**7. Connect an agent.** Serve over HTTP with a key, and point an MCP client at it:
[docs/serving.md](docs/serving.md).

```bash
export DECISION_SERVER_KEY=<a long random key>
uv run python server.py my_scenario.toml --http
```

**The server acts without asking.** When a decision calls for an acting tool, the tool runs
at once, unless the route is uncertain. Calling `decide` twice with the same request acts twice.
Point acting tools at real systems with care. Confirmation before acting is planned; the
scenario and the server's replies are already shaped for it.

Read [docs/data-handling.md](docs/data-handling.md) before any real text goes in.

## The server's tools

| MCP tool | What it does |
|---|---|
| `decide(text)` | The route with its confidence and probabilities, `uncertain` below the threshold, each question's answer, the rules that apply, the facts the informing tools found, and what the acting tools did |
| `check_call(text, call)` | CLM's opinion of a tool call the caller proposes: a probability and approved yes or no |
| `describe()` | The scenario's routes, questions and rules |
| `correct(text, route)` | Review mode only: save the right route for a request |
| `agreement()` | Review mode only: how often CLM's own route agrees with the saved corrections |

## What to expect

Measured on the example scenario, on an M5 Max with 64 GB:

- **Untrained, CLM routes poorly.** On held-back generated examples the released heads got
  51%; trained on 199 generated examples, 98%. Through the server, the trained heads routed 15
  of 17 hand-written test messages correctly. Seventeen messages is a small test.
- **CLM's probabilities are relative, not calibrated.** Treat the confidence threshold as a
  setting to tune with `evaluate.py`, not as a probability of being right.
- **The questions are not trained.** Only routing is. Check the answers to your questions by
  eye before relying on them.
- **Decisions are quick.** The slow test loads CLM and makes those 17 decisions, with their
  tool calls, in under 3 seconds.
- **The two backends agree.** With the released heads, the torch backend (on the Mac's GPU)
  and the MLX build chose the same route for 79 of 81 requests from the example. Trained with
  the torch backend, the example also reached 98% and 15 of 17. CLM has not yet been run on an
  NVIDIA GPU: see [docs/models.md](docs/models.md#what-has-been-checked).

## Layout

| File | Purpose |
|---|---|
| `scenario.py` | Reads and checks a scenario file |
| `decision.py` | One decision: route, informing tools, questions, rules, acting tools |
| `server.py` | The decision server: MCP over stdio or HTTP |
| `chat.py` | The test chat, an MCP client |
| `clm_router.py` | Loads CLM with either backend; routing and the reuse of corrections |
| `clm_torch.py`, `clm_schema.py`, `clm_common.py` | CLM with PyTorch; the authors' question format, copied unchanged; what both share |
| `tools.py` | Loads a scenario's tools |
| `record.py`, `evaluate.py` | The corrections record and its replay |
| `dataset.py`, `synthesize.py`, `train.py`, `generator.py` | Training, and the language model that writes examples |
| `examples/service_desk/` | The worked example, with a stand-in issue tracker |
| `tests/` | `uv run pytest` (no models needed); `uv run pytest -m slow` (loads CLM; `CLM_TEST_BACKEND=torch` for the torch backend) |
| `Dockerfile` | The project on Linux, for the torch backend |

CLM is described in Kwok et al. (2026) and published at
[Contrastive-LM/CLM](https://github.com/Contrastive-LM/CLM). This project runs the authors'
released model with PyTorch, or the community MLX builds of it on a Mac.

## Licence

Copyright 2026 Kevin Long. Licensed under the [Apache License 2.0](LICENSE). The licence covers this
repository only: the models it downloads have their own terms.
