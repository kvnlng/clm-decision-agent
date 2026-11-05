# Examples

Six examples: two scenarios and four environments ([getting started](../docs/getting-started.md#your-environment-and-the-decisions-over-it)
explains the two kinds). Each README has the full story, how to run it and what happened when models were asked.

| Example | What it shows | Data | Needs |
|---|---|---|---|
| [service_desk](service_desk/README.md) (scenario) | Which team handles a request, how urgent, identifiable data or not: CLM routes a message, tools look up the team and open a ticket in a mock tracker. Run first, then copy | Made up | CLM, trained once first (under a minute): `uv run clm-train examples/service_desk/scenario.toml` |
| [crowdstrike](crowdstrike/README.md) (scenario) | How to help each passenger after the 19 July 2024 outage: thousands of decisions at once, CLM against language models on speed and accuracy, an agent's proposals checked, the day replayed on a GPU | Real US flights (BTS); passengers, fares, messages and policy made up | CLM; an Anthropic API key for the Claude rows of `compare_models.py` (skipped without); a GPU with torch for the replay's data-centre figures |
| [tariff](tariff/README.md) (environment) | Should Product Line X source from Region B, and what happens to Q3 margin and SLA times: a graph, DMN rules, levers ranked by an objective with every limit checked, a CLM judge of supplier news | Made up | Nothing for `sourcing`; CLM (the `clm` extra) for the news judge; Neo4j optional, after loading the graph |
| [new_site](new_site/README.md) (environment) | Can we open a site in an expensive area: rules an agent can explore, facts and estimates told apart, warnings and headroom at an estimate's range, approvals, a decision record for a financial officer | Real census income (ACS 2023); the company made up | Nothing |
| [noncompete](noncompete/README.md) (environment) | Does the standard non-compete hold in each state, and how to narrow it: statutes and 4,800 court opinions in PostgreSQL, searched, read and followed by citation; an answer's citations checked against what the tools returned. Not legal advice | Real statutes and opinions (Caselaw Access Project); the company made up | PostgreSQL with pgvector; Ollama with `embeddinggemma` and `gemma4:26b`; CLM for its two judgements |
| [handbook](handbook/README.md) (environment) | The 401(k) match and four more employee questions over private documents no model has seen: a made-up handbook in PostgreSQL, rules that cite it, and what each layer adds (no tools; rules only; rules and handbook; handbook alone; rules derived by a model, checked, read again by another model and approved), counting wrong figures stated as fact. Not HR, tax or legal advice | Made up; the 2026 IRS limit from IRS Notice 2025-67 | PostgreSQL with pgvector; Ollama with `embeddinggemma`; an Anthropic API key to derive with Claude (`derived/` is committed, so that variant runs without one) |

`clm-agent` answers with Claude by default (an Anthropic API key or `ant auth login`) or a local model with
`--backend ollama`. The repository includes no trained heads and no runs: `runs/` folders and `*.heads/` are written
locally and ignored by git.

## Where to start

- **A scenario:** service_desk.
- **An environment:** tariff, then new_site; neither needs training.
  `npx -y @modelcontextprotocol/inspector@2.13.0 uv run clm-server examples/new_site/environment.toml` shows what
  an agent sees, in the browser ([Looking inside the server](../docs/getting-started.md#looking-inside-the-server)).
- **Your own:** `uv run clm-new-environment my_environment` writes a working environment to edit: a team choosing
  where to hold an offsite. [docs/environment.md](../docs/environment.md) walks from a new folder to a registered
  server; [docs/walkthrough.md](../docs/walkthrough.md) builds a scenario step by step, with crowdstrike as the
  worked case.
