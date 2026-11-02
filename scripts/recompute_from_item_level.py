"""Recompute the main results from the released parsed answers only (no GPU, no raw outputs, no MedQA text).

Usage (from repo/):
  python -X utf8 scripts/recompute_from_item_level.py --results results
Rebuilds minimal records from results/item_level.csv (parsed letters, truncation counts, token counts, GPU seconds),
runs the same evaluation code with the same seeds, and compares every row of results/results.csv. The greedy-rerun
rows (subset "rerun_subset") need the raw rerun outputs, which are not released, and are skipped.
Exit code 0 if every compared value matches exactly.
"""
import argparse
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from board_vote.config import load_config  # noqa: E402
from board_vote.evaluate import evaluate  # noqa: E402

N_RUNS = 10


def text(letter) -> str:
    return f"Answer: {letter}" if isinstance(letter, str) and letter else "no answer given"


def records_from_item_level(t: pd.DataFrame) -> dict[str, list[dict]]:
    greedy, sampled = [], []
    for r in t.itertuples(index=False):
        greedy.append({"id": r.id, "run": 0, "text": text(r.greedy), "finish": "length" if r.greedy_truncated else "stop",
                       "new_tokens": int(r.greedy_tokens), "batch_seconds": float(r.greedy_gpu_s), "batch_items": 1})
        for j in range(N_RUNS):
            sampled.append({"id": r.id, "run": j, "text": text(getattr(r, f"run{j}")),
                            "finish": "length" if j < int(r.sampled_truncated) else "stop",      # only the count is used
                            "new_tokens": int(r.sampled_tokens) if j == 0 else 0,               # only the sum is used
                            "batch_seconds": float(r.sampled_gpu_s), "batch_items": 1})
    return {"greedy": greedy, "sampled": sampled}


def same(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            a, b = float(a), float(b)
        except (TypeError, ValueError):
            return False
        return (math.isnan(a) and math.isnan(b)) or math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
    return str(a) == str(b)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()
    res = Path(a.results)
    t = pd.read_csv(res / "item_level.csv", keep_default_na=False, na_values=[""])
    items = t[["id", "key", "code", "objective_subset"]].copy()
    items["objective_subset"] = items["objective_subset"].astype(bool)
    _, new = evaluate(items, records_from_item_level(t), load_config(), n_boot=a.n_boot)
    old = pd.read_csv(res / "results.csv")
    old = old[old["subset"] != "rerun_subset"]
    merged = old.merge(new, on=["metric", "subset"], how="left", suffixes=("_old", "_new"))
    bad = []
    for r in merged.itertuples(index=False):
        for col in ("value", "ci_low", "ci_high", "n"):
            if not same(getattr(r, f"{col}_old"), getattr(r, f"{col}_new")):
                bad.append(f"{r.metric} / {r.subset} / {col}: released {getattr(r, f'{col}_old')} vs recomputed {getattr(r, f'{col}_new')}")
    print(f"{len(merged)} result rows compared (rerun rows skipped); {len(bad)} differences")
    for line in bad[:20]:
        print("  " + line)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
