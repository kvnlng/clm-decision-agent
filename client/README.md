# Test client

**For testing only.** Not for production, and not for real sensitive text. It is not part of the
`clm-decision-agent` package and is not installed with it.

A small command-line chat client for MCP. Point it at a running decision server, built from an example or your own
scenario or environment, and ask questions. A model answers from the server's tools, and every tool call and its
result are printed, so you can see where an answer came from. Whether the answers are good enough is yours to judge.
Use it when you have no MCP client such as Claude, or would rather not use one.

## Running it

From a checkout, with [uv](https://docs.astral.sh/uv/) and, for the starter model, [Ollama](https://ollama.com):

```bash
ollama pull granite4:micro                                            # the starter model, about 2.1 GB
export DECISION_SERVER_KEY=<a long random key>
uv run clm-server examples/tariff/environment.toml --http             # terminal 1: the server
uv run --project client clm-test-client http://127.0.0.1:8000/mcp     # terminal 2: the client
```

```
you> Who supplies the components of Product Line X?
  [tool: query_components_of {"product": "PX"}]
  [result: ...]
Product Line X has three components: ...
```

The client sends the key in `DECISION_SERVER_KEY` (a shared key or an OAuth token) as a bearer token and never
prints it. Follow-up questions keep the conversation. `/tools` lists the tools; `/quit` leaves. Results over 1,500
characters are shortened on screen (the model gets them whole); `--full` prints them in full.

## Any model

| Backend | Option | Model's key |
|---|---|---|
| Anything that speaks the OpenAI chat API: Ollama, LM Studio, vLLM, most hosted models | `--model NAME --model-url URL` (default: `granite4:micro` on Ollama) | `LLM_API_KEY`, if it needs one |
| Claude, through the Anthropic API | `uv run --project client --extra claude clm-test-client URL --backend claude [--model NAME]` | Anthropic's usual credentials |

The model must support tool calls. Read the printed tool results, not only the answer.

**The starter model**, `granite4:micro`, was chosen over `qwen3:4b` and `llama3.2:3b`: it was the fastest, and the
only one of the three to call `decide`. Against the same servers, three questions, once each:

| Question | `granite4:micro`, local, free | `claude-opus-5-5`, paid |
|---|---|---|
| Who supplies Product Line X's components? | Right | Right |
| Should sourcing move to Region B? | Passed the product's name, not its id `PX`; every branch failed | Right: 50%, and why 100% was rejected |
| Service desk: who handles a registry request? | Right route and ticket, but rewrote the ticket's address | Right, and did not claim more than the result showed |

What this showed:

- **A weak answer may be the model, not your server.** Every miss was the small model's; on the same tools, Claude
  answered all three. Try a larger model before changing your tools.
- **A small model shows where descriptions are unclear.** Two of the small models passed a component's name where
  its id was wanted; saying "give the id, such as C1" in the tariff descriptions fixed that for `granite4:micro`.
  Testing with a small model as well as a large one finds such gaps.
- **Speed is similar:** 4 to 7 seconds a question for `granite4:micro` on a Mac, about 10 for Claude.

## Dry runs

**By default every tool call is carried out**, including tools that act, so you can test your configuration fully.
Point acting tools at a test system if you do not want them to act for real.

- `--dry-run` prints every call and sends none.
- `--dry-run TOOL`, repeatable, holds back only those tools; the others run. A name the server does not have stops
  the client at start-up, so a typo cannot let a tool run.

A held-back call is printed as `[dry run, not sent: TOOL {arguments}]`, and the model is told it was not carried out.

In a scenario, acting tools run inside `decide`, so holding back `decide` skips the whole decision: you see that it
was called, not the route. To see decisions without real actions, point the acting tools at a stand-in, as the
service desk's `mock_tracker.py` does.

## Tests

`uv run --project client pytest client/tests` runs against a stand-in server and model; it needs no network or key.
