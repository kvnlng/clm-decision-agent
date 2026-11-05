# Scenario reference

The reference for [Writing a scenario](scenario.md): the key tables, the exact rules, the reply fields, the skipped
reasons and the measured figures, in the same order as the overview.

## Routes

| Key | Type | Default | Meaning |
|---|---|---|---|
| `[routing] instructions` | string | required | The question CLM is asked |
| `[routing] confidence_threshold` | 0 to 1 | `0.6` | Below this the decision is uncertain and no action runs |
| `[routing] inform` | tool names | `[]` | Run before routing; the route hears their facts ([below](#facts-the-route-considers)) |
| `[routes.<name>] description` | string | required | What CLM reads: full sentences about the person's need, not labels. Its wording matters more than anything else in the file |
| `[routes.<name>] examples` | strings | `[]` | Example messages |
| `[routes.<name>] inform` | tool names | `[]` | The route's informing tools, in order |
| `[routes.<name>.options]` | table | none | A choice fetched at decision time ([below](#options-fetched-at-decision-time)) |

## Questions for CLM

| Key | Meaning |
|---|---|
| `type` | `"noul"`: yes or no, answered 0 to 1. `"score"`: ordered levels, answered with the expected level. `"choice"`: one of several options |
| `instructions` | The question CLM is asked |
| `criteria` | `score`: at least two levels, lowest first. `choice`: a table of at least two options with descriptions. Not used by `noul` |

`clm-train` trains only questions your examples label ([fine-tuning.md](fine-tuning.md)); check the others by eye
before relying on them.

## Rules

| Key | Meaning |
|---|---|
| `when` | `<question> >= <number>` or `<= <number>` for `noul` and `score` questions; `<question> == <option>` for `choice` questions |
| `tell` | The guidance |

## Tools

| Key | Default | Meaning |
|---|---|---|
| `[tools] module` | none | The module that defines `TOOLS`, a list of functions, and optionally `configure(settings, scenario_dir)`, called once at start-up. It is loaded from the file beside the scenario, so no other module of that name can stand in for it. A name with no file there is imported as an installed package |
| `[tools.settings]` | `{}` | Passed to `configure()` |
| `[tools.inform.<tool>] questions` | all questions | Which questions hear the tool's facts: facts can sway unrelated questions |

Give each tool a mock backend as well as the live one, switched by a setting such as `backend = "mock"`, to train and
check before connecting real systems; `clm-new-scenario` writes tools that way
([walkthrough.md](walkthrough.md#4-act-on-it-acting-tools)). Make clients and connections once, in `configure()`: an
HTTP client made on every call can cost more than CLM's part of the decision. Tools are called from several threads
at once, so anything they share must allow that, as an `httpx.Client` does.

### Facts the route considers

- The reply's `route`, `confidence` and `probabilities` are the decision made with the facts; `before_facts` holds
  the route from the message alone, and is absent without `[routing] inform`.
- Their facts also go to the questions, as `[tools.inform]` says. A route that lists the same tool does not run it
  again.
- Corrections are saved, reused and trained on with the facts included, since that is the text the route saw.
- `describe` lists these tools as `route_facts_from`.

### Options fetched at decision time

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

| Key | Default | Meaning |
|---|---|---|
| `tool` | required | The function to run |
| `routes` | required, at least one | The routes it runs on |
| `when` | none | In the rules' syntax; the tool runs only when it holds |
| `confirm` | `false` | Only `false` for now: the tool runs without asking |

Every reply lists each action's status: `done` with the tool's result, `failed` with its message, or `skipped` with
the reason (`route uncertain`, `no option`, `option uncertain`, `read-only server`).

## Checking a caller's tool call

| Key | Default | Meaning |
|---|---|---|
| `[tool_check] instructions` | `"Is this tool call appropriate for what the user asked?"` | The question CLM is asked |
| `[tool_check] threshold` | `0.5` | At or above this (0 to 1) the call is approved |

CLM's opinion is advice. Training covers the routes and questions, not this question. On the CrowdStrike example it
approved the right call and refused the wrong one for 114 of 200. That run was on an M5 Max with trained heads,
approving at 0.5, over the 200 check cases whose answer is Rebook or Refund, with a rebooking and a refund proposed
for each ([examples/crowdstrike](../examples/crowdstrike/README.md), [findings.md](findings.md)).

## Corrections and reuse

`correct` saves the owner's route for a request to the record, optionally with the right answers
(`"answers": {"urgency": 2, "identifiable": true}`) and, on a route with options, the right option's id. The route's
informing tools run again, so send the same `context` as with `decide` and the tools will find the same facts.
`agreement` reports how often CLM's own route agrees with the saved corrections. A route reused from a correction is
never uncertain.

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
