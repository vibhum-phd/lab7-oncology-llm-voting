"""CPU tests for module 4 (evaluation). Every expected number below was worked out by hand first.

Part A: each statistic against a known answer. Part B: a designed mock data set of 100 items whose
metrics are known exactly. No real test-set result is used (none exists yet).
Usage (from repo/): python -I -X utf8 tests/test_evaluate.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from board_vote.config import load_config  # noqa: E402
from board_vote.evaluate import auroc, evaluate, fleiss_kappa, mcnemar_exact, paired_bootstrap_ci  # noqa: E402

results = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    results.append(bool(ok))


def close(a, b, tol=1e-3):
    return abs(a - b) <= tol


def make_mock():
    """100 items: 0-29 both right, unanimous; 30-54 both right (7/3 split); 55-59 greedy right, vote wrong;
    60-74 greedy wrong, vote right; 75-99 both wrong. Item 0 greedy truncated; item 99 greedy unparseable."""
    L = "ABCD"
    keys = [L[k % 4] for k in range(100)]
    wrong = [L[(k + 1) % 4] for k in range(100)]
    items = pd.DataFrame({"id": [f"m{k:03d}" for k in range(100)], "key": keys,
                          "code": ["M" if k % 5 == 0 else "O" for k in range(100)],
                          "objective_subset": [k % 2 == 0 for k in range(100)]})
    txt = lambda x: f"some reasoning\nAnswer: {x}" if x else "no answer given"
    greedy, sampled = [], []
    for k in range(100):
        g = keys[k] if k < 60 else wrong[k]
        if k == 99:
            g = None
        greedy.append({"id": items.id[k], "run": 0, "text": txt(g), "finish": "length" if k == 0 else "stop",
                       "new_tokens": 100, "batch_seconds": 10.0, "batch_items": 50})
        if k < 30:
            runs = [keys[k]] * 10
        elif k < 55 or 60 <= k < 75:                    # vote right: key in runs 0-6
            runs = [keys[k]] * 7 + [wrong[k]] * 3
        else:                                           # vote wrong: wrong in runs 0-6, key in 7-9
            runs = [wrong[k]] * 7 + [keys[k]] * 3
        for j, x in enumerate(runs):
            sampled.append({"id": items.id[k], "run": j, "text": txt(x), "finish": "stop", "new_tokens": 90,
                            "batch_seconds": 20.0, "batch_items": 50})
    rerun = [dict(r) for r in greedy[:50]]
    rerun[3]["text"] = rerun[3]["text"].replace("some reasoning", "slightly different reasoning")
    return items, {"greedy": greedy, "sampled": sampled, "greedy_rerun": rerun}


def main() -> int:
    # A. Statistics against known answers.
    wiki = np.array([[0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0], [2, 2, 8, 1, 1],
                     [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2], [6, 5, 2, 1, 0], [0, 2, 2, 3, 7]])
    check("Fleiss' kappa = 0.210 on the published worked example", close(fleiss_kappa(wiki), 0.210),
          f"{fleiss_kappa(wiki):.4f}")
    check("Fleiss' kappa = 1 when all raters agree", close(fleiss_kappa(np.array([[10, 0], [0, 10], [10, 0]])), 1.0))
    check("AUROC = 0.75 on a 2+2 example", auroc([0.9, 0.8, 0.7, 0.6], [1, 0, 1, 0]) == 0.75)
    check("AUROC = 0.5 when scores tie", auroc([0.5, 0.5], [1, 0]) == 0.5)
    check("McNemar exact p for b=15, c=5 = 0.0414", close(mcnemar_exact(15, 5), 0.0414, 1e-4), f"{mcnemar_exact(15, 5):.5f}")
    check("McNemar p = 1 with no discordant items", mcnemar_exact(0, 0) == 1.0)
    x, y = np.r_[np.ones(70), np.zeros(30)], np.r_[np.ones(60), np.zeros(40)]
    ci1, ci2 = paired_bootstrap_ci(x, y, 2000, 1), paired_bootstrap_ci(x, y, 2000, 1)
    check("bootstrap CI is reproducible with a fixed seed and contains the estimate",
          ci1 == ci2 and ci1[0] <= 0.10 <= ci1[1], f"{ci1}")

    # B. Designed mock data set.
    cfg = load_config()
    items, records = make_mock()
    t, res = evaluate(items, records, cfg, n_boot=2000)
    v = lambda m, s="all": res.loc[(res.metric == m) & (res.subset == s), "value"].iloc[0]
    note = lambda m, s="all": res.loc[(res.metric == m) & (res.subset == s), "note"].iloc[0]
    check("accuracy greedy 0.60, majority 0.70", close(v("accuracy_greedy"), 0.60) and close(v("accuracy_majority10"), 0.70))
    check("primary difference 0.10", close(v("accuracy_difference_majority10_minus_greedy"), 0.10))
    check("McNemar uses b=15, c=5", "b=15" in note("accuracy_difference_majority10_minus_greedy")
          and "c=5" in note("accuracy_difference_majority10_minus_greedy") and "p=0.04139" in note("accuracy_difference_majority10_minus_greedy"),
          note("accuracy_difference_majority10_minus_greedy"))
    check("lenient analysis drops the unparseable item (n = 99)",
          res.loc[(res.metric == "accuracy_greedy") & (res.subset == "all (lenient)"), "n"].iloc[0] == 99)
    check("single-sample mean accuracy 0.67", close(v("accuracy_single_sample_mean"), 0.67))
    check("majority of first 1 = 0.70 and of 10 = 0.70",
          close(v("accuracy_majority_of_first_1"), 0.70) and close(v("accuracy_majority_of_first_10"), 0.70))
    check("unanimity 0.30, mean modal share 0.79", close(v("unanimity_rate"), 0.30) and close(v("mean_modal_share"), 0.79))
    check("pairwise flip rate 0.3267", close(v("pairwise_flip_rate"), 0.70 * 21 / 45))
    check("AUROC 0.7143", close(v("auroc_modal_share_vs_majority_correct"), 1500 / 2100))
    check("accuracy at >=9/10 agreement 1.0, below 0.5714",
          close(v("accuracy_agreement_ge9"), 1.0) and close(v("accuracy_agreement_lt9"), 40 / 70))
    check("unparseable greedy 0.01, truncation greedy 0.01",
          close(v("unparseable_rate_greedy"), 0.01) and close(v("truncation_rate_greedy"), 0.01))
    check("greedy rerun: letters 1.00, text 0.98",
          close(v("greedy_rerun_identical_letter", "rerun_subset"), 1.0) and close(v("greedy_rerun_identical_text", "rerun_subset"), 0.98))
    check("GPU seconds summed per item share", close(v("gpu_seconds_greedy"), 100 * 10.0 / 50) and close(v("gpu_seconds_sampled"), 100 * 20.0 / 50))
    check("subgroups and objective subset reported",
          {"subgroup_M", "subgroup_O", "objective_subset"} <= set(res.subset))
    check("item table has no text column", not any("text" in c for c in t.columns))

    # C. Refuses incomplete or unknown outputs.
    bad = {k: list(r) for k, r in records.items()}
    bad["sampled"] = [r for r in bad["sampled"] if not (r["id"] == "m010" and r["run"] == 4)]
    try:
        evaluate(items, bad, cfg, n_boot=10)
        check("an item missing one sampled run is refused", False)
    except ValueError as e:
        check("an item missing one sampled run is refused", True, str(e))

    print(f"{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
