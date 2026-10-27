"""Start a new scenario: a folder with a scenario file, a tools module and labelled cases to check it on.

    uv run python new_scenario.py FOLDER

The folder holds a small working scenario for a shop's customer messages, to edit into your own:

    scenario.toml           routes, a question, a rule, an informing and an acting tool
    <folder>_tools.py       order_status (informing) and open_case (acting), each with a mock backend
                            and a live one to write, switched by [tools.settings] backend
    scenario.cases.jsonl    labelled cases: `evaluate.py --cases` scores the scenario on them

Then: `synthesize.py` writes training examples, `train.py` trains the heads, `evaluate.py --cases` checks
them and `server.py` serves the decisions. docs/walkthrough.md goes through it step by step.
"""
import argparse
import json
import platform
import re
import sys
from pathlib import Path

SCENARIO = '''\
# A scenario for a shop's customer messages: the server decides what the customer needs, looks up their
# order and opens a case for a refund. Edit it into your own; docs/scenario.md explains every part.
# Run with the context {{"order": "<order id>"}}: the tools find the order from it.

# Keep one of the two blocks below and comment out the other.
[clm]
{clm}
heads = "scenario.heads"            # trained heads, made by train.py; the released ones are used until then

# Only synthesize.py uses a language model, to write example messages for training
[generator]
{generator}
max_tokens = 500

[synthesize]
messages = "messages that customers might send to an online shop's help desk"

[routing]
instructions = "What does the customer need?"
confidence_threshold = 0.6          # below this the decision is marked uncertain and no action runs
inform = ["order_status"]           # run first: the route is decided with the order's facts

[tools]
module = "{module}"                 # {module}.py beside this file

[tools.settings]
backend = "mock"                    # "live" once its functions in {module}.py are written

# The order's facts go to the route, not to the question of how urgent the customer is
[tools.inform.order_status]
questions = []

[routes.Refund]
description = "The customer wants their money back for an order."
examples = ["The kettle arrived broken, I want a refund"]

[routes.Where_Is_It]
description = "The customer is asking where their order is or when it will arrive."
examples = ["My parcel was due on Monday and still hasn't come"]

# CLM always picks one of the routes, so a scenario needs one for requests it does not cover.
[routes.Other]
description = "Anything else, or a message too vague to tell what the customer needs."
examples = ["Hi, quick question"]

[questions.urgent]
type = "noul"
instructions = "Does the customer need this sorted out urgently?"

[[rules]]
when = "urgent >= 0.5"
tell = "Answer today and say so."

# Runs as soon as the decision is made, unless it is uncertain
[[actions]]
tool = "open_case"
routes = ["Refund"]
'''

CLM_MLX = '''\
# On a Mac with Apple Silicon: the MLX build
backend = "mlx"
model_id = "mlx-community/CLM-v0.1-8B-MLX-8bit"
# On Linux with an NVIDIA GPU (or any machine, slowly, on the CPU): the authors' release with PyTorch
# backend = "torch"
# model_id = "Contrastive-LM/CLM-v0.1-8B"'''

CLM_TORCH = '''\
# On a Mac with Apple Silicon: the MLX build
# backend = "mlx"
# model_id = "mlx-community/CLM-v0.1-8B-MLX-8bit"
# On Linux with an NVIDIA GPU (or any machine, slowly, on the CPU): the authors' release with PyTorch
backend = "torch"
model_id = "Contrastive-LM/CLM-v0.1-8B"'''

GENERATOR_MLX = '''\
backend = "mlx"                                        # "mlx", "ollama" or "openai"
model = "mlx-community/Qwen3-4B-Instruct-2507-4bit"'''

GENERATOR_OLLAMA = '''\
backend = "ollama"                                     # "mlx" (on a Mac), "ollama" or "openai"
model = "gemma4:26b"                                   # any model your Ollama server has'''

TOOLS = '''\
"""Tools for the {folder} scenario. The decision server calls each one with the request:
{{"text", "route", "answers", "rules", "facts", "context"}}, filled in as far as the decision has got.

    order_status    an informing tool: the order's facts, for the route's question
    open_case       an acting tool: opens a case for a refund

Each has a mock backend, which works now, and a live one for you to write: [tools.settings] backend
switches between them, so the scenario can be trained and checked before the shop's systems are connected.
"""

SETTINGS: dict = {{}}

# The mock's orders; the live backend asks the shop's order system instead
ORDERS = {{
    "A100": {{"item": "a kettle", "status": "delivered", "days_ago": 2}},
    "A101": {{"item": "a desk lamp", "status": "in transit", "days_late": 3}},
}}
CASES: list[dict] = []      # the cases the mock has opened


def configure(settings: dict, scenario_dir) -> None:  # pylint: disable=unused-argument
    """Take the scenario's [tools.settings]. Make clients and connections here, once, not in every call."""
    SETTINGS.clear()
    SETTINGS.update(settings)
    if SETTINGS.get("backend", "mock") not in ("mock", "live"):
        raise ValueError("backend must be mock or live")


def _live() -> bool:
    return SETTINGS.get("backend", "mock") == "live"


def order_status(request: dict) -> dict:
    """Facts about the customer's order, as plain sentences: the tool works things out, CLM judges."""
    order_id = (request.get("context") or {{}}).get("order")
    if order_id is None:
        return {{"facts": ["No order number was given."]}}
    if _live():
        raise NotImplementedError("order_status has no live backend yet: write it in this file")
    order = ORDERS.get(order_id)
    if order is None:
        return {{"facts": [f"There is no order {{order_id}}."]}}
    if order["status"] == "delivered":
        return {{"facts": [f"Order {{order_id}}, {{order['item']}}, was delivered {{order['days_ago']}} days ago."]}}
    return {{"facts": [f"Order {{order_id}}, {{order['item']}}, is {{order['status']}}, "
                      f"{{order['days_late']}} days late."]}}


def open_case(request: dict) -> dict:
    """Open a refund case. Put every safeguard here: the server runs this whenever the decision says so."""
    order_id = (request.get("context") or {{}}).get("order")
    if order_id is None:
        raise ValueError("no order number in the context, so no case was opened")
    if _live():
        raise NotImplementedError("open_case has no live backend yet: write it in this file")
    case = {{"case": len(CASES) + 1, "order": order_id, "route": request["route"]}}
    CASES.append(case)
    return case


TOOLS = [order_status, open_case]
'''

CASES = [
    {"text": "The kettle I ordered arrived with a cracked lid. I'd like my money back please.",
     "context": {"order": "A100"}, "route": "Refund", "answers": {"urgent": False}},
    {"text": "Refund order A100 today, I need the money for rent!!", "context": {"order": "A100"},
     "route": "Refund", "answers": {"urgent": True}},
    {"text": "Where is my lamp? It should have been here days ago.", "context": {"order": "A101"},
     "route": "Where_Is_It", "answers": {"urgent": False}},
    {"text": "I need the lamp for an event tomorrow and there's no sign of it. When will it come?",
     "context": {"order": "A101"}, "route": "Where_Is_It", "answers": {"urgent": True}},
    {"text": "Do you sell gift cards?", "route": "Other", "answers": {"urgent": False}},
    {"text": "hello?? anyone there", "route": "Other", "answers": {"urgent": False}},
]


def mlx_available() -> bool:
    """Whether this machine can run the MLX build: a Mac with Apple Silicon."""
    return sys.platform == "darwin" and platform.machine() == "arm64"


def write(folder: Path) -> Path:
    """Write the starter files into `folder`; -> the scenario file."""
    module = re.sub(r"\W", "_", folder.name.lower()).strip("_") + "_tools"
    if module[0].isdigit():
        module = "s_" + module
    mlx = mlx_available()
    folder.mkdir(parents=True, exist_ok=True)
    scenario = folder / "scenario.toml"
    scenario.write_text(SCENARIO.format(clm=CLM_MLX if mlx else CLM_TORCH,
                                        generator=GENERATOR_MLX if mlx else GENERATOR_OLLAMA, module=module),
                        encoding="utf-8")
    (folder / f"{module}.py").write_text(TOOLS.format(folder=folder.name), encoding="utf-8")
    (folder / "scenario.cases.jsonl").write_text("".join(json.dumps(case) + "\n" for case in CASES),
                                                 encoding="utf-8")
    return scenario


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description="Start a new scenario in a folder of its own.")
    parser.add_argument("folder", type=Path, help="the folder to create; it must be new or empty")
    args = parser.parse_args(argv)
    if args.folder.exists() and not args.folder.is_dir():
        print(f"error: {args.folder} is a file; choose a new folder", file=sys.stderr)
        return 2
    if args.folder.exists() and any(args.folder.iterdir()):
        print(f"error: {args.folder} is not empty; choose a new folder", file=sys.stderr)
        return 2
    scenario = write(args.folder)
    print(f"Wrote {scenario}, its tools and its cases. Next:\n"
          f"  uv run python synthesize.py {scenario}        # training examples\n"
          f"  uv run python train.py {scenario}\n"
          f"  uv run python evaluate.py {scenario} --cases {args.folder / 'scenario.cases.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
