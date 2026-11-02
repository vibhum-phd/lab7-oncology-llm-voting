"""Exploratory analyses from protocol v2 (amendments_v2). CPU only. Kept apart from the frozen RQ1 results.

Usage (from repo/):
  python -X utf8 scripts/run_extras_evaluation.py --split test --raw results/raw/test --out results
Writes <out>/results_v2_exploratory.csv with: greedy self-agreement (greedy vs greedy_rerun_full),
the temperature comparison (T = 0.3, 0.7 = main run, 1.0), and the contamination-probe match rate.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from board_vote.config import REPO_ROOT, load_config  # noqa: E402
from board_vote.data import load_medqa, read_ids, select_items  # noqa: E402
from board_vote.evaluate import evaluate, mcnemar_exact, paired_bootstrap_ci  # noqa: E402
from board_vote.parse import AnswerParser  # noqa: E402
from board_vote.plots import fig_temperature  # noqa: E402
from board_vote.probe import probe_messages  # noqa: E402
from board_vote.probe import exact_match  # noqa: E402

TEMPERATURE_ARMS = [("sampled_t03", 0.3), ("sampled", 0.7), ("sampled_t10", 1.0)]
KEEP = ["accuracy_majority10", "accuracy_difference_majority10_minus_greedy", "accuracy_single_sample_mean",
        "unanimity_rate", "mean_modal_share", "fleiss_kappa_across_runs", "pairwise_flip_rate",
        "auroc_modal_share_vs_majority_correct", "unparseable_rate_sampled", "truncation_rate_sampled"]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def row(analysis, metric, value, ci=(np.nan, np.nan), n=None, note=""):
    return {"analysis": analysis, "metric": metric, "value": value, "ci_low": ci[0], "ci_high": ci[1], "n": n, "note": note}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["debug", "test"], required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "protocol_v2.yaml"))
    ap.add_argument("--raw-dir", default=str(ROOT / "data" / "raw" / "medqa_4opt_hf"))
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()

    cfg = load_config(a.config)
    if "amendments_v2" not in cfg:
        print("refusing: needs protocol_v2.yaml")
        return 2
    parse, seed, raw = AnswerParser(cfg["parsing"]), cfg["model"]["engine_args"]["seed"], Path(a.raw)
    ids = sorted(read_ids(ROOT / "data" / "splits" / f"{a.split}_ids.txt"))
    all_items = {it.id: it for it in select_items(load_medqa(a.raw_dir), ids)}
    info = pd.read_csv(ROOT / "data" / "splits" / "items_final.csv").set_index("id").loc[ids]
    items = pd.DataFrame({"id": ids, "key": [all_items[i].key for i in ids], "code": info["code"].to_list(),
                          "objective_subset": info["objective_subset"].astype(bool).to_list()})
    greedy = read_jsonl(raw / "greedy.jsonl")
    rows = []

    # 1. Greedy self-agreement on all items.
    rerun = read_jsonl(raw / "greedy_rerun_full.jsonl")
    if rerun:
        g = {r["id"]: parse(r["text"]) for r in greedy}
        h = {r["id"]: parse(r["text"]) for r in rerun}
        if set(g) != set(h) or set(g) != set(ids):
            raise ValueError("greedy and greedy_rerun_full do not cover the same items")
        key = dict(zip(items["id"], items["key"]))
        x = np.array([h[i] == key[i] for i in ids], float)
        y = np.array([g[i] == key[i] for i in ids], float)
        b, c = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
        same = np.mean([g[i] == h[i] for i in ids])
        rows += [row("greedy_self_agreement", "same_letter_rate", same, n=len(ids)),
                 row("greedy_self_agreement", "accuracy_greedy", y.mean(), n=len(ids)),
                 row("greedy_self_agreement", "accuracy_greedy_rerun_full", x.mean(), n=len(ids)),
                 row("greedy_self_agreement", "accuracy_difference_rerun_minus_greedy", x.mean() - y.mean(),
                     paired_bootstrap_ci(x, y, a.n_boot, seed), len(ids),
                     f"McNemar exact p={mcnemar_exact(b, c):.4g}; b={b}, c={c} (noise floor for RQ1; exploratory)")]

    # 2. Temperature comparison (exploratory, no hypothesis).
    for arm, temp in TEMPERATURE_ARMS:
        recs = read_jsonl(raw / f"{arm}.jsonl")
        if not recs:
            continue
        _, res = evaluate(items, {"greedy": greedy, "sampled": recs}, cfg, n_boot=a.n_boot)
        res = res[(res["subset"] == "all") & (res["metric"].isin(KEEP))]
        for r in res.itertuples(index=False):
            rows.append(row(f"temperature_{temp}", r.metric, r.value, (r.ci_low, r.ci_high), r.n,
                            (r.note if isinstance(r.note, str) else "") + " (exploratory)"))

    # 3. Contamination probe.
    probe = read_jsonl(raw / "probe.jsonl")
    if probe:
        hits = [exact_match(r["text"], probe_messages(all_items[r["id"]])[1]) for r in probe]
        k, n = int(sum(hits)), len(hits)
        ci = binomtest(k, n).proportion_ci(method="wilson") if n else None
        rows.append(row("contamination_probe", "exact_match_rate", k / n if n else np.nan,
                        (ci.low, ci.high) if ci else (np.nan, np.nan), n,
                        f"{k}/{n} hidden options reproduced exactly (normalised); exploratory"))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.insert(0, "split", a.split)
    df.to_csv(out / "results_v2_exploratory.csv", index=False)
    print(f"{len(df)} exploratory rows -> {out / 'results_v2_exploratory.csv'}")
    if rerun and all(f"temperature_{t}" in set(df["analysis"]) for _, t in TEMPERATURE_ARMS):
        print("figure:", ", ".join(p.name for p in fig_temperature(df, out / "figures")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
