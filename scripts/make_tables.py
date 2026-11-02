"""Result tables for the paper, built only from the evaluation outputs (Prompt 13). CPU only, no item text.

Usage (from repo/):
  python -X utf8 scripts/make_tables.py --results results
Reads <results>/results.csv, item_level.csv and (if present) results_v2_exploratory.csv; writes
<results>/tables/table{1..5}_*.csv and <results>/tables/tables.md. Every cell names its source row below the table.
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

pct = lambda x: f"{100 * x:.1f}"
ci = lambda lo, hi: f"[{100 * lo:.1f}, {100 * hi:.1f}]"


def markdown(df: pd.DataFrame) -> str:
    cells = [list(map(str, df.columns))] + [[str(x) for x in r] for r in df.itertuples(index=False)]
    lines = ["| " + " | ".join(c) + " |" for c in cells]
    return "\n".join([lines[0], "|" + "---|" * len(df.columns)] + lines[1:])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    d = Path(a.results)
    r = pd.read_csv(d / "results.csv")
    t = pd.read_csv(d / "item_level.csv")
    ex = pd.read_csv(d / "results_v2_exploratory.csv") if (d / "results_v2_exploratory.csv").exists() else None

    def row(metric, subset="all", frame=r, key="subset"):
        hit = frame[(frame["metric"] == metric) & (frame[key] == subset)]
        if len(hit) != 1:
            raise ValueError(f"expected one row for {metric} / {subset}, found {len(hit)}")
        return hit.iloc[0]

    def mcnemar(note):
        m = re.search(r"p=([0-9.eE-]+); discordant b=(\d+).*c=(\d+)", note) or re.search(r"p=([0-9.eE-]+); b=(\d+), c=(\d+)", note)
        return float(m.group(1)), int(m.group(2)), int(m.group(3))

    tables = {}

    # Table 1. RQ1 (confirmatory) and its sensitivity analyses.
    rows = []
    for subset, label in (("all", "All test items (primary)"), ("all (lenient)", "Lenient: unparseable greedy items dropped"),
                          ("objective_subset", "Objective subset (oncology term in key)"),
                          ("subgroup_M", "Subgroup M (management)"), ("subgroup_O", "Subgroup O (other oncology)")):
        g, v = row("accuracy_greedy", subset), row("accuracy_majority10", subset)
        dd = row("accuracy_difference_majority10_minus_greedy", subset)
        p, b, c = mcnemar(dd["note"])
        rows.append({"Analysis": label, "n": int(dd["n"]), "Greedy % [95% CI]": f"{pct(g.value)} {ci(g.ci_low, g.ci_high)}",
                     "Majority-of-10 % [95% CI]": f"{pct(v.value)} {ci(v.ci_low, v.ci_high)}",
                     "Difference, points [95% CI]": f"{100 * dd.value:+.1f} {ci(dd.ci_low, dd.ci_high)}",
                     "Vote only right (b)": b, "Greedy only right (c)": c, "McNemar exact p": f"{p:.3g}",
                     "Frozen rule (p < 0.05 and CI > 0)": "met" if p < 0.05 and dd.ci_low > 0 else "not met"})
    tables["table1_rq1_accuracy"] = (pd.DataFrame(rows),
        "RQ1: majority vote over 10 sampled runs (T = 0.7) vs greedy decoding, paired on the same items.",
        "results.csv rows accuracy_greedy, accuracy_majority10, accuracy_difference_majority10_minus_greedy for each subset; "
        "b, c and p from the note of the difference row. Only the first row is confirmatory.")

    # Table 2. RQ2 consistency and RQ3 agreement as a reliability signal (main run, T = 0.7).
    rows = []
    for metric, label in (("unanimity_rate", "Unanimity (10/10 runs give the same letter)"),
                          ("mean_modal_share", "Mean modal share"),
                          ("fleiss_kappa_across_runs", "Fleiss' kappa across the 10 runs"),
                          ("pairwise_flip_rate", "Pairwise flip rate (of 45 run pairs)"),
                          ("auroc_modal_share_vs_majority_correct", "AUROC: modal share vs vote correct")):
        x = row(metric)
        scale = 100 if metric in ("unanimity_rate", "pairwise_flip_rate") else 1
        fmt = (lambda y: f"{y:.1f}") if scale == 100 else (lambda y: f"{y:.3f}")
        rows.append({"Measure": label, "Value [95% CI]": f"{fmt(scale * x.value)} [{fmt(scale * x.ci_low)}, {fmt(scale * x.ci_high)}]",
                     "n items": int(x.n)})
    for metric, label in (("accuracy_agreement_ge9", "Vote accuracy when >= 9/10 runs agree"),
                          ("accuracy_agreement_lt9", "Vote accuracy when < 9/10 runs agree")):
        x = row(metric)
        rows.append({"Measure": label, "Value [95% CI]": f"{pct(x.value)} %", "n items": int(x.n)})
    x = row("coverage_at_agreement_ge10")
    acc10 = float(re.search(r"= ([0-9.]+)", x.note).group(1))
    rows.append({"Measure": "Items with 10/10 agreement: coverage; vote accuracy among them",
                 "Value [95% CI]": f"{pct(x.value)} %; {pct(acc10)} %", "n items": int(x.n)})
    tables["table2_rq2_rq3_consistency"] = (pd.DataFrame(rows),
        "RQ2 (run-to-run consistency) and RQ3 (agreement as a reliability signal), 10 sampled runs at T = 0.7.",
        "results.csv rows unanimity_rate, mean_modal_share, fleiss_kappa_across_runs (2,000 bootstrap resamples), "
        "pairwise_flip_rate, auroc_modal_share_vs_majority_correct (2,000), accuracy_agreement_ge9/lt9, "
        "coverage_at_agreement_ge10. Unanimity and flip rate in %.")

    # Table 3. Cost next to performance.
    gt, st = row("output_tokens_total_greedy").value, row("output_tokens_total_sampled").value
    gs, ss = row("gpu_seconds_greedy").value, row("gpu_seconds_sampled").value
    n = int(row("accuracy_greedy").n)
    single = row("accuracy_single_sample_mean")
    rows = [
        {"Method": "Greedy (1 run, T = 0)", "Accuracy %": pct(row("accuracy_greedy").value),
         "Output tokens": int(gt), "Tokens per item": f"{gt / n:.0f}", "GPU seconds": f"{gs:.0f}", "Relative tokens": "1.0x",
         "Relative GPU time": "1.0x"},
        {"Method": "One sampled run (T = 0.7), mean of 10", "Accuracy %": pct(single.value),
         "Output tokens": int(round(st / 10)), "Tokens per item": f"{st / 10 / n:.0f}", "GPU seconds": "n/a (generated together)",
         "Relative tokens": f"{st / 10 / gt:.1f}x", "Relative GPU time": "n/a"},
        {"Method": "Majority of 10 sampled runs (T = 0.7)", "Accuracy %": pct(row("accuracy_majority10").value),
         "Output tokens": int(st), "Tokens per item": f"{st / n:.0f}", "GPU seconds": f"{ss:.0f}", "Relative tokens": f"{st / gt:.1f}x",
         "Relative GPU time": f"{ss / gs:.1f}x"},
    ]
    tables["table3_cost"] = (pd.DataFrame(rows),
        "Compute cost beside accuracy (633 test items, Kaggle T4 x2, vLLM 0.32.0, FP16).",
        "results.csv rows output_tokens_total_*, gpu_seconds_* (engine wall time summed over batches), accuracy_*; "
        f"single-run accuracy: {single.note}. The 10 sampled runs are generated in one request per item, so a "
        "single run's GPU time is not separable.")

    # Table 4. Agreement pattern of greedy and vote, item by item (from item_level.csv).
    runs = t[[f"run{j}_correct" for j in range(10)]].to_numpy(bool)
    all_wrong = int((~t["greedy_correct"].astype(bool) & ~runs.any(axis=1)).sum())
    all_right = int((t["greedy_correct"].astype(bool) & runs.all(axis=1)).sum())
    gc, vc = t["greedy_correct"].astype(bool), t["vote_correct"].astype(bool)
    rows = [{"Pattern": "Both right", "Items": int((gc & vc).sum())},
            {"Pattern": "Vote right, greedy wrong (b)", "Items": int((vc & ~gc).sum())},
            {"Pattern": "Greedy right, vote wrong (c)", "Items": int((gc & ~vc).sum())},
            {"Pattern": "Both wrong", "Items": int((~gc & ~vc).sum())},
            {"Pattern": "Right in all 11 generations (greedy + 10 runs)", "Items": all_right},
            {"Pattern": "Wrong in all 11 generations (greedy + 10 runs)", "Items": all_wrong},
            {"Pattern": "Vote decided by the tie rule (earliest run)", "Items": int(t["vote_tie"].astype(bool).sum())}]
    tables["table4_item_patterns"] = (pd.DataFrame(rows),
        "Item-level agreement pattern, 633 test items (T = 0.7).",
        "item_level.csv columns greedy_correct, vote_correct, run0..9_correct, vote_tie. "
        "An unparseable answer counts as wrong.")

    # Table 5. Exploratory runs from protocol v2 (no hypotheses; not part of the confirmatory test).
    if ex is not None:
        e = lambda analysis, metric: row(metric, analysis, ex, "analysis")
        rows = []
        for temp in ("0.3", "0.7", "1.0"):
            an = f"temperature_{temp}"
            dd = e(an, "accuracy_difference_majority10_minus_greedy")
            p, b, c = mcnemar(dd["note"])
            rows.append({"Setting": f"Majority of 10 at T = {temp}" + (" (main run)" if temp == "0.7" else ""),
                         "Accuracy %": pct(e(an, "accuracy_majority10").value),
                         "Single-run mean %": pct(e(an, "accuracy_single_sample_mean").value),
                         "Difference vs greedy, points [95% CI]": f"{100 * dd.value:+.1f} {ci(dd.ci_low, dd.ci_high)}",
                         "McNemar p (b, c)": f"{p:.3g} ({b}, {c})",
                         "Unanimity %": pct(e(an, "unanimity_rate").value),
                         "Fleiss' kappa": f"{e(an, 'fleiss_kappa_across_runs').value:.3f}",
                         "AUROC": f"{e(an, 'auroc_modal_share_vs_majority_correct').value:.3f}"})
        sd = e("greedy_self_agreement", "accuracy_difference_rerun_minus_greedy")
        p, b, c = mcnemar(sd["note"])
        rows.append({"Setting": "Greedy rerun of all items (noise floor)",
                     "Accuracy %": pct(e("greedy_self_agreement", "accuracy_greedy_rerun_full").value),
                     "Single-run mean %": "",
                     "Difference vs greedy, points [95% CI]": f"{100 * sd.value:+.1f} {ci(sd.ci_low, sd.ci_high)}",
                     "McNemar p (b, c)": f"{p:.3g} ({b}, {c})",
                     "Unanimity %": f"same letter {pct(e('greedy_self_agreement', 'same_letter_rate').value)}",
                     "Fleiss' kappa": "", "AUROC": ""})
        pr = e("contamination_probe", "exact_match_rate")
        rows.append({"Setting": "Contamination probe (hidden option reproduced exactly)",
                     "Accuracy %": f"exact match {pct(pr.value)} {ci(pr.ci_low, pr.ci_high)} (Wilson, n = {int(pr.n)})",
                     "Single-run mean %": "", "Difference vs greedy, points [95% CI]": "", "McNemar p (b, c)": "",
                     "Unanimity %": "", "Fleiss' kappa": "", "AUROC": ""})
        tables["table5_exploratory_v2"] = (pd.DataFrame(rows),
            "Exploratory analyses added by protocol v2 before the comparison was viewed (temperature, greedy noise floor, "
            "contamination probe). Not confirmatory; no correction for multiple comparisons.",
            "results_v2_exploratory.csv (analysis, metric). The T = 0.7 row repeats the main run.")

    out = d / "tables"
    out.mkdir(parents=True, exist_ok=True)
    md = ["# Result tables (generated by scripts/make_tables.py; do not edit by hand)\n"]
    for k, (name, (df, title, source)) in enumerate(tables.items(), 1):
        df.to_csv(out / f"{name}.csv", index=False)
        md += [f"## Table {k}. {title}\n", markdown(df), f"\n*Source:* {source}\n"]
    (out / "tables.md").write_text("\n".join(md), encoding="utf-8")
    print(f"{len(tables)} tables -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
