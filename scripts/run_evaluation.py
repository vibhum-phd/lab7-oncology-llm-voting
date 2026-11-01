"""Evaluate saved outputs for one split (module 4) and draw the figures (module 5b). CPU only.

Usage (from repo/):
  python -X utf8 scripts/run_evaluation.py --split debug --raw results/raw/debug --out results/debug
  python -X utf8 scripts/run_evaluation.py --split test  --raw results/raw/test  --out results
Writes <out>/results.csv (metrics), <out>/item_level.csv (letters and flags, no text), <out>/figures/.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from board_vote.config import load_config  # noqa: E402
from board_vote.data import load_medqa, read_ids, select_items  # noqa: E402
from board_vote.evaluate import evaluate  # noqa: E402
from board_vote.plots import make_all  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["debug", "test"], required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw-dir", default=str(ROOT / "data" / "raw" / "medqa_4opt_hf"))
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()

    cfg = load_config()
    ids = sorted(read_ids(ROOT / "data" / "splits" / f"{a.split}_ids.txt"))
    keys = {it.id: it.key for it in select_items(load_medqa(a.raw_dir), ids)}
    info = pd.read_csv(ROOT / "data" / "splits" / "items_final.csv").set_index("id").loc[ids]
    items = pd.DataFrame({"id": ids, "key": [keys[i] for i in ids], "code": info["code"].to_list(),
                          "objective_subset": info["objective_subset"].astype(bool).to_list()})
    raw = Path(a.raw)
    records = {arm: read_jsonl(raw / f"{arm}.jsonl") for arm in ("greedy", "sampled", "greedy_rerun")}
    table, res = evaluate(items, records, cfg, n_boot=a.n_boot)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res.insert(0, "split", a.split)
    res.to_csv(out / "results.csv", index=False)
    table.to_csv(out / "item_level.csv", index=False)
    figs = make_all(res, table, cfg["decoding"]["sampled"]["n"], out / "figures")
    print(f"{len(res)} metric rows -> {out / 'results.csv'}; {len(table)} items -> {out / 'item_level.csv'}")
    print("figures:", ", ".join(p.name for p in figs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
