"""How CLM's delivery-risk question separates labelled news lines: run before choosing news_risk's threshold.

    uv run python examples/tariff/judge_news.py

It loads CLM through the example's [clm] settings, asks the delivery_risk judge of each line in news_cases.jsonl
and prints each probability, then the lowest among the risks against the highest among the others. CLM's
probabilities are relative, not calibrated, so a threshold is only worth choosing where the two groups separate.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from engine import run_judgement  # pylint: disable=wrong-import-position
from environment import load_environment  # pylint: disable=wrong-import-position


def measure(env, cases: list[dict]) -> list[tuple[float, dict]]:
    """Each case with CLM's probability that it is a delivery risk."""
    return [(run_judgement(env, "delivery_risk", {"text": case["text"]})["result"]["probability"], case)
            for case in cases]


def separation(judged: list[tuple[float, dict]]) -> tuple[float, float, float | None]:
    """-> (lowest probability of a risk, highest of the others, the threshold midway, or None if they overlap)."""
    risks = [p for p, case in judged if case["risk"]]
    others = [p for p, case in judged if not case["risk"]]
    low, high = min(risks), max(others)
    return low, high, round((low + high) / 2, 3) if low > high else None


def main() -> int:
    """Print the table and the separation."""
    env = load_environment(HERE / "environment.toml")
    cases = [json.loads(line) for line in (HERE / "news_cases.jsonl").read_text(encoding="utf-8").splitlines()]
    judged = measure(env, cases)
    for probability, case in sorted(judged, key=lambda item: -item[0]):
        print(f"{probability:.3f}  {'risk ' if case['risk'] else 'other'}  {case['text'][:90]}")
    low, high, threshold = separation(judged)
    print(f"\nLowest risk {low:.3f}, highest other {high:.3f}: "
          + (f"a threshold of {threshold} separates them" if threshold else "no threshold separates them"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
