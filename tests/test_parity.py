"""The torch backend against the MLX build, on the same requests with the released heads.

Run on an Apple Silicon Mac with: uv run pytest -m slow -k parity -s

The MLX build is the authors' model at 8 bits; its publishers measured 99% agreement with the authors'
own server. The torch backend runs the authors' model at full precision, so the two should mostly agree.
"""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring
import json
from pathlib import Path

import pytest
from desk_routing_cases import CASES

from clm_router import LocalContrastiveRouter, load_engine
from scenario import load_scenario

pytestmark = pytest.mark.slow
pytest.importorskip("mlx.core")

EXAMPLE = Path(__file__).parent.parent / "examples" / "service_desk"


def test_torch_and_mlx_agree_on_the_example():
    scenario = load_scenario(EXAMPLE / "scenario.toml")
    examples = [json.loads(line) for line in (EXAMPLE / "scenario.examples.jsonl").read_text().splitlines()]
    texts = [text for text, _ in CASES] + [example["text"] for example in examples[::5]]
    decided = {}
    for backend, model_id in (("mlx", "mlx-community/CLM-v0.1-8B-MLX-8bit"), ("torch", "Contrastive-LM/CLM-v0.1-8B")):
        router = LocalContrastiveRouter(engine=load_engine(backend, model_id))
        decided[backend] = [router.decide(text, scenario) for text in texts]
    agree = sum(m["route"] == t["route"] for m, t in zip(decided["mlx"], decided["torch"]))
    largest = max(abs(m["probabilities"][name] - t["probabilities"][name])
                  for m, t in zip(decided["mlx"], decided["torch"]) for name in scenario.routes)
    print(f"\ntorch and mlx agree on the route for {agree} of {len(texts)} requests; "
          f"the largest route probability difference is {largest:.3f}")
    assert agree >= 0.9 * len(texts)
