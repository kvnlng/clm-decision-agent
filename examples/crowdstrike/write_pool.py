"""Ask a local model for the passengers' message lines, bucket by bucket, into message_pool.jsonl.

    uv run python examples/crowdstrike/write_pool.py [--per-bucket 40] [--buckets N]

Run once; the pool is committed. Lines that don't say what their bucket needs are discarded
(messages.fits). Buckets already in the file are skipped, so a stopped run carries on where it was.
"""
import os
# Set before huggingface_hub is imported, which reads it at import time
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# pylint: disable=wrong-import-position

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path


from messages import BUCKETS, POOL_PATH, fits  # noqa: E402  # pylint: disable=wrong-import-order

from clm_decision.generator import MLXGenerator  # noqa: E402
from clm_decision.synthesize import parse_messages  # noqa: E402

MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
TEMPERATURE = 0.8            # sampled rather than greedy: greedy output repeats the prompt's own words
STYLES = ("short and to the point", "polite, two or three sentences", "written in a hurry, lower case",
          "a question", "starts with what happened before asking", "rambling, with a little background")
WANT = {
    "travel": "They still want to get to {dest}, on another flight.",
    "refund": "They have given up on the trip and want a refund (they use the word refund).",
    "vague": ("They say something is wrong and they don't know what to do, and nothing more: not whether they "
              "still want to travel, nor anything about money. They never write rebook, another flight, next "
              "flight, refund or train."),
}
DEADLINE = {
    "none": "They have nowhere they need to be by any particular time, and mention no event.",
    "preferred": ("However they feel, they say plainly that they would like to make {purpose} but could live "
                  "with missing it; they never write must, have to or can't miss."),
    "firm": "They cannot miss {purpose}, so they must be in {dest} by {deadline}, and they say both.",
    "-": "They mention no event and no time they need to arrive by.",
}
VULNERABLE = {"vulnerable": "{companion} is travelling with them, and they mention it.",
              "-": ("Nobody with them needs special care; they mention no children, elderly, disabled or pregnant "
                    "travellers.")}
MODE = {"air": "They don't mention trains.", "rail": "They would rather go by train than fly, and say so.",
        "either": "A train or a flight would both do for them, and they say so.", "-": "They don't mention trains."}
UPSET = {"calm": "calm and polite", "annoyed": "clearly annoyed", "angry": "furious, perhaps rude"}
PROMPT = (
    "Write {count} messages that different airline passengers typed to the airline on a day their flight was "
    "cancelled or badly delayed. Each passenger writes in their own words; don't reuse the wording of these "
    "notes, and make every message different from the others.\n"
    "About these passengers:\n- {want}\n- {deadline}\n- {vulnerable}\n- {mode}\n- They are {upset}.\n"
    "Style: {style}.\n"
    "Write these placeholders exactly, with their braces, instead of real details; they are filled in later: "
    "{{flight}} the flight number, {{city}} where they are, {{dest}} where they're going, {{time}} when it was "
    "due to leave, {{party}} who is travelling (like \"the two of us\"){extra}. Use no other braces, and "
    "never name an event or a person yourself: write {{purpose}} or {{companion}}.\n"
    "Return only a JSON list of strings."
)

class SampledGenerator(MLXGenerator):
    """The local model, sampling with a fixed seed so a rerun writes the same pool."""

    def __init__(self, model_id: str, max_tokens: int, seed: int = 0):
        super().__init__(model_id, max_tokens)
        import mlx.core as mx  # pylint: disable=import-outside-toplevel,import-error
        from mlx_lm.sample_utils import make_sampler  # pylint: disable=import-outside-toplevel,import-error
        mx.random.seed(seed)
        self.sampler = make_sampler(temp=TEMPERATURE, top_p=0.95)

    def generate(self, system: str, user: str) -> str:
        """The model's reply, sampled."""
        from mlx_lm import generate  # pylint: disable=import-outside-toplevel,import-error
        return generate(self.model, self.tokenizer, prompt=self.build_prompt(system, user),
                        max_tokens=self.max_tokens, sampler=self.sampler, verbose=False).strip()


def build_prompt(key: str, style: str, count: int) -> str:
    """The request for one batch of lines for one bucket."""
    reason, deadline, vulnerable, mode, upset = key.split("|")
    extra = {"firm": ", {deadline} the time they must arrive by, {purpose} the event they must not miss",
             "preferred": ", {purpose} the event they hope to make"}
    extra_text = extra.get(deadline, "") + (", {companion} who travels with them" if vulnerable == "vulnerable" else "")
    return PROMPT.format(count=count, want=WANT[reason], deadline=DEADLINE[deadline], vulnerable=VULNERABLE[vulnerable],
                         mode=MODE[mode], upset=UPSET[upset], style=style, extra=extra_text)


def fill(generator, key: str, per_bucket: int, per_call: int = 8, write=print) -> tuple[list[str], Counter]:
    """Lines for one bucket, until it has enough or five calls in a row bring nothing new."""
    kept: list[str] = []
    dropped: Counter = Counter()
    call, idle = 0, 0
    while len(kept) < per_bucket and idle < 5 and call < 6 * per_bucket // per_call:
        reply = generator.generate(build_prompt(key, STYLES[call % len(STYLES)], per_call), f"Batch {call + 1}.")
        fresh = []
        # A reply that isn't quite JSON can leave its opening bracket on the first line
        for line in dict.fromkeys(line.lstrip("[").strip().strip('"') for line in parse_messages(reply)):
            if line in kept:
                continue
            if fits(key, line):
                fresh.append(line)
            else:
                dropped["doesn't fit"] += 1
        kept += fresh[:per_bucket - len(kept)]
        idle = 0 if fresh else idle + 1
        call += 1
    write(f"  {key:<36}{len(kept):>4} kept, {sum(dropped.values()):>3} dropped, {call} calls")
    return kept, dropped


# Written by hand: the model would not write {purpose} into a sentence this short, and invented events instead
CLAUSES = {
    "calm": ["We'd like to make {purpose} if possible, but we can live with missing it.",
             "It would be nice to get there in time for {purpose}, though it isn't essential.",
             "We were hoping to make {purpose}, but it's not the end of the world.",
             "Ideally we'd make {purpose}, but we could live without it."],
    "annoyed": ["And yes, we'd hoped to make {purpose}; fine, we can live with missing it.",
                "We were hoping to make {purpose}, which now looks unlikely, but whatever.",
                "It would be nice to make {purpose}, not that anyone here seems to care.",
                "I'd like to still make {purpose}, though I suppose we could live without it."],
    "angry": ["We were hoping to make {purpose} too. Not essential, but it would be nice if you managed anything.",
              "I'd like to make {purpose}, though at this rate I can live with missing it.",
              "Ideally we'd still make {purpose}. It isn't essential, but you've ruined enough already.",
              "We'd love to make {purpose}, but frankly I could live with missing it more than with this airline."],
}
LEAST = 20                   # lines a bucket needs; short preferred buckets are topped up by composing


def compose(key: str, pool: dict[str, list[str]], made: list[str], have: int, rng) -> list[str]:
    """Lines for a short preferred bucket: a line of the same bucket without a deadline, then a clause."""
    base = pool.get(key.replace("|preferred|", "|none|"), [])
    found: list[str] = []
    for _ in range(200):
        if have + len(found) >= LEAST or not base or not made:
            break
        line = f"{rng.choice(base)} {rng.choice(made)}"
        if line not in found and fits(key, line):
            found.append(line)
    return found


def top_up(out: Path, write=print) -> None:
    """Compose lines for the preferred buckets the model could not fill."""
    import random  # pylint: disable=import-outside-toplevel
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]
    pool: dict[str, list[str]] = {}
    for line in lines:
        pool.setdefault(line["bucket"], []).append(line["text"])
    rng = random.Random(0)
    for key in BUCKETS:
        if "|preferred|" not in key or len(pool.get(key, [])) >= LEAST:
            continue
        found = compose(key, pool, CLAUSES[key.split("|")[-1]], len(pool.get(key, [])), rng)
        with open(out, "a", encoding="utf-8") as file:
            file.writelines(json.dumps({"bucket": key, "text": line, "composed": True}) + "\n" for line in found)
        write(f"  {key:<36}{len(found):>4} composed")


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description="Write the CrowdStrike passengers' message pool.")
    parser.add_argument("--per-bucket", type=int, default=40)
    parser.add_argument("--buckets", type=int, default=len(BUCKETS), help="only the first N buckets (a pilot)")
    parser.add_argument("--out", type=Path, default=POOL_PATH)
    args = parser.parse_args(argv)
    done = set()
    if args.out.exists():
        done = {json.loads(line)["bucket"] for line in args.out.read_text(encoding="utf-8").splitlines() if line}
    generator = SampledGenerator(MODEL, 600)
    start = time.perf_counter()
    for key in BUCKETS[:args.buckets]:
        if key in done:
            continue
        kept, _ = fill(generator, key, args.per_bucket)
        with open(args.out, "a", encoding="utf-8") as file:
            file.writelines(json.dumps({"bucket": key, "text": line}) + "\n" for line in kept)
    top_up(args.out)
    print(f"done in {time.perf_counter() - start:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
