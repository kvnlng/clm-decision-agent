"""How CLM's two questions separate passages labelled from the courts' own words: run before trusting either judge.

    uv run python examples/noncompete/judge_labels.py

Each label in labels/broader.jsonl and labels/departs.jsonl is a passage of an opinion in the cases collection, with
`yes` true where the court said so: that the covenant was broader than needed, or that it declined to follow or
criticised the case it cites. The script first checks that every label's text is in the passage it names, then asks
the judge of each text and prints each probability, and the lowest among the yeses against the highest among the
noes. CLM's probabilities are relative, not calibrated, so a threshold is only worth choosing where the two separate.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

from clm_decision.engine import Steps, run_judgement  # pylint: disable=wrong-import-position
from clm_decision.environment import load_environment  # pylint: disable=wrong-import-position

LABELS = {"broader_than_needed": "broader.jsonl", "departs": "departs.jsonl"}


def unquoted(env, labels: list[dict]) -> list[dict]:
    """The labels whose text is not in the passage of the opinion they name."""
    steps = Steps(env)
    missing = []
    for label in labels:
        passage = steps.document("cases", label["document"], label["passage"])["passages"][0]["text"]
        if " ".join(label["text"].split()) not in " ".join(passage.split()):
            missing.append(label)
    return missing


def measure(env, judge: str, labels: list[dict]) -> list[tuple[float, dict]]:
    """Each label with CLM's probability for the judge's question."""
    return [(run_judgement(env, judge, {"text": label["text"]})["result"]["probability"], label) for label in labels]


def separation(judged: list[tuple[float, dict]]) -> tuple[float, float, float | None]:
    """-> (lowest probability of a yes, highest of a no, the threshold midway, or None if they overlap)."""
    low = min(p for p, label in judged if label["yes"])
    high = max(p for p, label in judged if not label["yes"])
    return low, high, round((low + high) / 2, 3) if low > high else None


def main() -> int:
    """Print each judge's table and separation; exit 1 if any label misquotes its passage."""
    env = load_environment(HERE / "environment.toml")
    for judge, name in LABELS.items():
        labels = [json.loads(line) for line in (HERE / "labels" / name).read_text(encoding="utf-8").splitlines()]
        missing = unquoted(env, labels)
        if missing:
            for label in missing:
                print(f"{name}: not in {label['document']} passage {label['passage']}: {label['text'][:80]}")
            return 1
        judged = measure(env, judge, labels)
        print(f"\n{judge} ({len(labels)} labels)")
        for probability, label in sorted(judged, key=lambda item: -item[0]):
            answer = "yes" if label["yes"] else "no "
            print(f"{probability:.3f}  {answer}  {label['document']:<12} {label['text'][:80]}")
        low, high, threshold = separation(judged)
        print(f"Lowest yes {low:.3f}, highest no {high:.3f}, a gap of {low - high:+.3f}: "
              + (f"a threshold of {threshold} separates them" if threshold else "no threshold separates them"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
