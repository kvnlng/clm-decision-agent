# Service desk demo

A front end to a data and analytics service desk. Someone working in a clinical environment
describes what they need; the assistant works out what kind of request it is, asks for the
missing details, drafts the ticket and, when you say yes, submits it to a tracker.

It shows the pattern the framework is built around: **classify, collect, act, report.**
It is not a clinical tool. All data here is invented.

## Run it

Two terminals, from the project root.

```bash
uv run python examples/service_desk/mock_tracker.py                      # terminal 1: the mock tracker
uv run python examples/service_desk/mock_openmetadata.py                 # terminal 2: the mock catalogue
uv run python chat.py examples/service_desk/scenario.toml --review     # terminal 3: the assistant
```

Try: `I need access to the oncology registry`, then answer its questions. Type `draft` to see
what it has collected, `new` to start over, `help` for the request types.

Without `--review` the assistant does not pause for your verdict on its decisions, but it
still asks before it submits a ticket.

## What decides what

| Decision | Made by |
|---|---|
| The kind of request, on the first message | CLM; you can correct it in review mode, and the correction is reused |
| Which dataset is meant | CLM, from the catalogue in OpenMetadata; the language model asks when CLM is not sure |
| How urgent it is, and whether identifiable data is involved | CLM, informed by what the catalogue knows about the dataset; the rules in the scenario turn these into instructions for the ticket text |
| Which details are still missing | Code, from the fields listed for the request type |
| The wording of questions and of the ticket | The language model |
| Whether to submit | You |

A request with fields to collect must have a tool listed under `confirm` in `[tools]`: that
tool succeeding is what completes the request.

A request in progress keeps its type. A reply such as "about three months" is not
re-classified.

## The files

| File | What it is |
|---|---|
| `scenario.toml` | Request types, the fields each needs, CLM's extra questions, the rules, and the tool settings |
| `service_desk_tools.py` | The tools. Each reaches a different kind of data source |
| `queues.json` | Which team handles each request type (read by `lookup_queue`) |
| `openmetadata_catalogue.py` | Reads the dataset catalogue from OpenMetadata and writes each table as an entry with facts (used by `list_datasets` and `find_dataset`) |
| `mock_openmetadata.py` | A stand-in for OpenMetadata's REST interface, serving eight invented datasets |
| `scenario.examples.jsonl` | Labelled example messages for training |
| `mock_tracker.py` | A stand-in for an issue tracker's REST interface (Forgejo's) |

## Training CLM for the scenario

As released, CLM routes first messages poorly: 8 of 20 hand-written test messages. One
command fits it to this scenario in under a minute:

```bash
uv run python train.py examples/service_desk/scenario.toml
```

That learns from `scenario.examples.jsonl` (318 example messages, written by the language
model and shipped here) and from any routes you have corrected in review mode. It saves the
result in `scenario.heads/`, which the chat picks up at its next start.

To write a fresh set of examples, delete `scenario.examples.jsonl` and run
`uv run python synthesize.py examples/service_desk/scenario.toml`.

What it did here (your numbers will differ by a message or two from run to run):

| | Released | Trained |
|---|---|---|
| Held-back generated examples (61) | 31% | 98% |
| Hand-written test messages (20) | 8 | 16 |

- The generated examples are easier than real messages, so the hand-written row is the one to
  believe, and 20 messages is a small test.
- On the hand-written messages, none of the wrong answers was confident enough to act on.
- The answers to the other questions (urgency, identifiable data) moved by about 0.03.
- **Change a route's description or the routing question and you must train again.** The chat
  notices, says so at start-up, and falls back to the released model until you do.
- Only the two small "heads" on top of CLM are trained; the large encoder is untouched.

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
6. In the terminal where you run the assistant: `export TRACKER_TOKEN=<the token>`.

The assistant then creates real issues, which you can see in the browser. Priority and other
tags become labels (`priority-high`, `governance-review`); a label the repository does not
have is created.

For a different tracker, rewrite the three tracker tools in `service_desk_tools.py`. Nothing
else depends on them.

## Using a real OpenMetadata instead of the mock

The catalogue comes from [OpenMetadata](https://open-metadata.org)'s REST API: the tables,
with their owners, tags and domains, and each table's lineage. OpenMetadata follows its own
graph to answer; the assistant only reads the answers, once, at start-up.

1. In OpenMetadata, create a bot account and copy its token.
2. In `[tools.settings]` set `openmetadata_url` to your instance, for example
   `"http://localhost:8585"`, and optionally `openmetadata_domain` to keep one domain's tables.
   CLM chooses between at most 50 datasets.
3. In the terminal where you run the assistant: `export OPENMETADATA_TOKEN=<the token>`.

**This has been run only against the mock.** The request paths and field names follow
OpenMetadata's documentation; check them against your version first (older versions, for
example, return `owner` rather than `owners`). Tables without a description are left out,
because CLM chooses by description: the better your descriptions, the better it chooses.

For another catalogue, rewrite `openmetadata_catalogue.py`, or the two dataset tools.
Nothing else depends on them.

## Pointing at other models

Both models are named in `scenario.toml`, with commented examples there.

- **Language model.** `backend = "mlx"` runs it in this process, `"ollama"` uses Ollama, and
  `"openai"` uses any server with the OpenAI chat interface (`base_url`, `model`, optional
  `api_key_env`). The demo has been run all three ways; the third against Ollama's
  OpenAI-style address with `gemma4:26b`.
- **CLM.** `backend = "server"` with a `url` asks the CLM authors' server instead of running
  CLM here. This follows the authors' server code but **has not been run against a real
  server**, which needs a GPU machine. With a server, a reviewed route is reused only when the
  conversation text is exactly the same, and `train.py` does not apply.

## Where text goes

At start-up the assistant prints where the text of a conversation goes: each model (this
process, a server on this computer, or another machine), and the record.

Only review mode saves anything. The record holds the text of reviewed conversations, so:

- it is created readable by you only;
- `path` under `[record]` moves it somewhere else;
- `uv run python record.py purge examples/service_desk/scenario.toml` deletes it, after asking;
- `key_file` or `key_env` under `[record]` encrypts each line. Make a key with
  `uv run python record.py newkey examples/service_desk/record.key` (a `key_file` is looked for
  beside the scenario). Keep the key out of version control and back it
  up: without it the record cannot be read.

Encryption protects the file on disk and nothing more. The text is still sent to whichever
models and tools the scenario names, anyone with the key can read the record, and the example
messages and trained heads are separate, unencrypted files. This is not de-identification, and
nothing here makes a deployment compliant with any regulation.

## Limitations

- **Routing is only reliable after training.** See "Training CLM for the scenario" above. Even
  then expect some misses; in the check above every miss had low confidence, so the assistant
  asked a clarifying question instead of acting on it.
- **Nothing stops someone typing patient details.** The prompts tell the model not to copy
  identifiers into a ticket, but that is an instruction, not a safeguard. The review record
  beside the scenario file stores the conversation text.
- **The details collected are a first guess** at what such a desk needs. Edit the `fields`
  tables in `scenario.toml`.
