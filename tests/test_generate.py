"""CPU tests for module 3 (resumable generation) with the mock engine. No GPU, no vLLM.

Usage (from repo/): python -I -X utf8 tests/test_generate.py
"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from board_vote.config import load_config  # noqa: E402
from board_vote.generate import MockEngine, item_seed, load_done, run_arm  # noqa: E402

results = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    results.append(bool(ok))


def read(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def main() -> int:
    cfg = load_config()
    dec = cfg["decoding"]
    base = cfg["model"]["engine_args"]["seed"]          # 20261102, the protocol's seed base
    ids = [f"item-{k:03d}" for k in range(25)]
    prompts = [f"prompt for {i} " * (k % 5 + 1) for k, i in enumerate(ids)]
    seeds = [item_seed(base, k) for k in range(len(ids))]

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # 1. Normal run.
        out = tmp / "greedy.jsonl"
        s = run_arm(MockEngine(dec), "greedy", ids, prompts, seeds, out, dec, batch_size=10)
        recs = read(out)
        check("greedy: 1 output per item", len(recs) == 25 and s["generated_now"] == 25, f"{s}")
        check("seed rule base + index recorded", [r["seed"] for r in recs] == seeds)

        # 2. Sampled arm writes 10 runs per item.
        out = tmp / "sampled.jsonl"
        run_arm(MockEngine(dec), "sampled", ids, prompts, seeds, out, dec, batch_size=10)
        recs = read(out)
        check("sampled: 10 outputs per item", len(recs) == 250 and
              all(sorted(r["run"] for r in recs if r["id"] == i) == list(range(10)) for i in ids))

        # 3. Interrupted run, then resume.
        out = tmp / "resume.jsonl"
        try:
            run_arm(MockEngine(dec, fail_after_batches=2), "sampled", ids, prompts, seeds, out, dec, batch_size=10)
        except RuntimeError:
            pass
        before = read(out)
        check("after the stop, 2 batches (20 items) are saved", len({r["id"] for r in before}) == 20)
        s = run_arm(MockEngine(dec), "sampled", ids, prompts, seeds, out, dec, batch_size=10)
        after = read(out)
        check("resume generates only the 5 missing items", s["already_done"] == 20 and s["generated_now"] == 5, f"{s}")
        check("no duplicates, all 25 complete", len(after) == 250 and len({(r["id"], r["run"]) for r in after}) == 250)
        check("saved outputs unchanged by resume", after[:200] == before)
        fresh = read(tmp / "sampled.jsonl")
        check("resumed file equals an uninterrupted run (same seeds)",
              sorted((r["id"], r["run"], r["text"]) for r in after) == sorted((r["id"], r["run"], r["text"]) for r in fresh))

        # 4. A half-written last line and an incomplete item are cleaned up.
        out = tmp / "broken.jsonl"
        good = read(tmp / "sampled.jsonl")
        lines = [json.dumps(r) for r in good[:30]]           # items 0-1 complete, item 2 has 10 runs; cut to 25
        lines = lines[:25] + ['{"id": "item-002", "arm": "sampled", "ru']   # item 2 incomplete + broken line
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        done = load_done(out, 10)
        check("broken line and incomplete item removed", done == {"item-000", "item-001"} and len(read(out)) == 20,
              f"done {sorted(done)}")

        # 5. Wrong output count from an engine is refused.
        class BadEngine(MockEngine):
            def generate(self, prompts, arm, seeds, max_tokens=None):
                return super().generate(prompts, arm, seeds, max_tokens)[:-1]
        try:
            run_arm(BadEngine(dec), "greedy", ids, prompts, seeds, tmp / "bad.jsonl", dec, batch_size=10)
            check("engine returning too few outputs refused", False)
        except RuntimeError:
            check("engine returning too few outputs refused", True)

    print(f"{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
