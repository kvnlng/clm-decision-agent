# Examples

Four examples, two of each kind of decision ([the README](../README.md#two-kinds-of-decision) explains the two
kinds). Each folder has its own README with the full story, how to run it and what happened when models were asked.

| Example | Kind | The question | What it shows | Data |
|---|---|---|---|---|
| [service_desk](service_desk/README.md) | Scenario | What kind of request is this, how urgent, does it involve identifiable data, and which team handles it? | The small finished scenario: CLM routes a message, tools look up the team and open a ticket in a mock tracker. The one to run first and copy | Made up |
| [crowdstrike](crowdstrike/README.md) | Scenario | How should we help each passenger after the 19 July 2024 outage? | The benchmark: thousands of decisions at once, CLM against language models on speed and accuracy, an agent's proposals checked, the day replayed on a GPU | Real US flights (BTS); passengers, fares, messages and policy made up |
| [tariff](tariff/README.md) | Environment | Should Product Line X move its sourcing to Region B, and what would it do to the Q3 margin and SLA times? | A graph, DMN rules, levers ranked by an objective with every limit checked; a CLM judge of supplier news | Made up |
| [new_site](new_site/README.md) | Environment | Are we in a position to open a new site in an expensive area? | Rules an agent can explore, facts and estimates told apart, warnings and headroom at an estimate's range, approvals, and a decision record for a financial officer | Real census income (ACS 2023); the company made up |

## Where to start

- **A scenario:** service_desk. CLM must be trained once first (under a minute):
  `uv run clm-train examples/service_desk/scenario.toml`.
- **An environment:** tariff, then new_site. Neither needs training; tariff's news judge needs the `clm` extra.
  `uv run clm-call examples/new_site/environment.toml rules '{"topic": "open_site"}'` shows what an agent sees.

The repository includes no trained heads and no runs: `runs/` folders and `*.heads/` are written locally and
ignored by git.

## Your own

`uv run clm-new-environment my_environment` writes a working environment to edit into your own: a team choosing
where to hold an offsite. [docs/environment.md](../docs/environment.md) walks from a new folder to a registered
server; [docs/walkthrough.md](../docs/walkthrough.md) builds a scenario step by step, with crowdstrike as the
worked case.
