"""Post hoc analyses requested by the peer review (Prompt 16). CPU only; writes no item text. Not pre-specified:
every row is descriptive and labelled post hoc in the paper.

Usage (from repo/):
  python -X utf8 scripts/posthoc_review.py --split test --raw results/raw/test --results results
Reads <results>/item_level.csv and the raw outputs; writes <results>/posthoc_review.csv and
<results>/figures/fig2b_majority_of_k_orders.{png,pdf} (majority-of-k averaged over random run orders).
Analyses: (1) RQ1 split into a sampling part and a voting part; vote against each single run; (2) RQ1 by MedQA split
of origin; (3) unanimity differences between temperatures; (4) comparison AUROCs for RQ3; (5) answer-position
counts and accuracy by key letter; (6) majority-of-k over 200 random run orders; (7) probe hits by option length and
origin (aggregate counts); (8) small corrections (b-items without unparseable greedy, observed discordance).
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from board_vote.config import load_config  # noqa: E402
from board_vote.data import load_medqa, read_ids, select_items  # noqa: E402
from board_vote.evaluate import auroc, mcnemar_exact, paired_bootstrap_ci  # noqa: E402
from board_vote.parse import AnswerParser, majority_vote  # noqa: E402
from board_vote.plots import BLUE, INK, INK2, ORANGE, _save, _style, plt  # noqa: E402
from board_vote.probe import exact_match, probe_messages  # noqa: E402

N_ORDERS, N_BOOT_SMALL = 200, 2000


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def letters_by_item(recs: list[dict], parse) -> dict[str, list]:
    by = {}
    for r in sorted(recs, key=lambda r: (r["id"], r["run"])):
        by.setdefault(r["id"], []).append(parse(r["text"]))
    return by


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["debug", "test"], required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--results", default="results")
    ap.add_argument("--raw-dir", default=str(ROOT / "data" / "raw" / "medqa_4opt_hf"))
    a = ap.parse_args()
    cfg = load_config()
    parse, seed, raw, res = AnswerParser(cfg["parsing"]), cfg["model"]["engine_args"]["seed"], Path(a.raw), Path(a.results)
    t = pd.read_csv(res / "item_level.csv", keep_default_na=False, na_values=[""])
    ids, key = t["id"].tolist(), t["key"].to_numpy(object)
    g = t["greedy_correct"].astype(bool).to_numpy().astype(float)
    v = t["vote_correct"].astype(bool).to_numpy().astype(float)
    runs = t[[f"run{j}_correct" for j in range(10)]].astype(bool).to_numpy()
    L = t[[f"run{j}" for j in range(10)]].astype(object).where(t[[f"run{j}" for j in range(10)]].notna(), None).to_numpy(object)
    greedy = t["greedy"].astype(object).where(t["greedy"].notna(), None).to_numpy(object)
    rows = []
    add = lambda group, metric, value, lo=np.nan, hi=np.nan, n=None, note="": rows.append(
        {"group": group, "metric": metric, "value": value, "ci_low": lo, "ci_high": hi, "n": n, "note": note})

    # 1. Sampling part and voting part of the RQ1 difference (item-level paired bootstrap, 10,000 resamples).
    single = runs.mean(axis=1)
    lo, hi = paired_bootstrap_ci(single, g, 10000, seed)
    add("decomposition", "single_run_mean_minus_greedy", single.mean() - g.mean(), lo, hi, len(t))
    lo, hi = paired_bootstrap_ci(v, single, 10000, seed)
    add("decomposition", "vote_minus_single_run_mean", v.mean() - single.mean(), lo, hi, len(t))
    per_run = []
    for j in range(10):
        r = runs[:, j].astype(float)
        b, c = int(((v == 1) & (r == 0)).sum()), int(((v == 0) & (r == 1)).sum())
        per_run.append((b, c, mcnemar_exact(b, c), v.mean() - r.mean()))
    bs, cs, ps, ds = zip(*per_run)
    add("decomposition", "vote_minus_each_run_min_to_max", min(ds), min(ds), max(ds), len(t),
        f"b {min(bs)}-{max(bs)}, c {min(cs)}-{max(cs)}, McNemar p {min(ps):.3g}-{max(ps):.3g} over the 10 runs")
    add("decomposition", "observed_discordance_rate", ((v != g).sum()) / len(t), n=len(t), note="(b + c) / n")

    # 2. RQ1 by MedQA split of origin (ID prefix train-/dev-/test-).
    origin = np.array([i.split("-")[0] for i in ids])
    for o in ("train", "dev", "test"):
        s = origin == o
        b, c = int(((v == 1) & (g == 0) & s).sum()), int(((v == 0) & (g == 1) & s).sum())
        add(f"origin_{o}", "n", int(s.sum()))
        add(f"origin_{o}", "accuracy_greedy", g[s].mean(), n=int(s.sum()))
        add(f"origin_{o}", "accuracy_vote", v[s].mean(), n=int(s.sum()))
        lo, hi = paired_bootstrap_ci(v[s], g[s], 10000, seed)
        add(f"origin_{o}", "difference", v[s].mean() - g[s].mean(), lo, hi, int(s.sum()), f"b={b}, c={c}, McNemar p={mcnemar_exact(b, c):.3g}")

    # 3. Like-for-like agreement and unanimity differences between temperatures (paired bootstrap over items).
    by_t = {0.7: {i: list(r) for i, r in zip(ids, L)}}
    for arm, temp in (("sampled_t03", 0.3), ("sampled_t10", 1.0)):
        recs = read_jsonl(raw / f"{arm}.jsonl")
        if recs:
            by_t[temp] = letters_by_item(recs, parse)
    unan = {tp: np.array([float(len(set(d[i])) == 1 and d[i][0] is not None) for i in ids]) for tp, d in by_t.items()}
    for tp in (0.3, 1.0):
        if tp in unan:
            lo, hi = paired_bootstrap_ci(unan[tp], unan[0.7], 10000, seed)
            add("temperature", f"unanimity_T{tp}_minus_T0.7", unan[tp].mean() - unan[0.7].mean(), lo, hi, len(t))
    if 0.3 in unan and 1.0 in unan:
        lo, hi = paired_bootstrap_ci(unan[1.0], unan[0.3], 10000, seed)
        add("temperature", "unanimity_T1.0_minus_T0.3", unan[1.0].mean() - unan[0.3].mean(), lo, hi, len(t))

    # 4. Comparison signals for RQ3 (bootstrap CIs, 2,000 resamples).
    def auc_ci(score, label, s):
        rng = np.random.default_rng(s)
        boots = []
        for _ in range(N_BOOT_SMALL):
            k = rng.integers(0, len(score), len(score))
            boots.append(auroc(score[k], label[k]))
        return auroc(score, label), np.nanpercentile(boots, 2.5), np.nanpercentile(boots, 97.5)
    first3 = np.array([majority_vote(list(r[:3])).count / 3 for r in L])
    ms = t["modal_share"].to_numpy(float)
    agree_greedy = np.array([float(gr is not None and gr == vt) for gr, vt in zip(greedy, t["vote"])])
    for name, score, label, s in (("modal_share_10_runs_vs_vote_correct", ms, v.astype(bool), seed + 7),
                                  ("modal_share_first_3_runs_vs_vote_correct", first3, v.astype(bool), seed + 8),
                                  ("greedy_equals_vote_vs_vote_correct", agree_greedy, v.astype(bool), seed + 9),
                                  ("modal_share_10_runs_vs_greedy_correct", ms, g.astype(bool), seed + 10)):
        val, lo, hi = auc_ci(score, label, s)
        add("rq3_comparison", name, val, lo, hi, len(t), f"bootstrap {N_BOOT_SMALL}")

    # 5. Answer position: letters chosen when wrong vs expected if wrong choices spread evenly; accuracy by key letter.
    def wrong_counts(pairs):
        obs = Counter(p for p, k in pairs if p is not None and p != k)
        exp = Counter()
        for p, k in pairs:
            if p is not None and p != k:
                for x in "ABCD":
                    if x != k:
                        exp[x] += 1 / 3
        return obs, exp
    og, eg = wrong_counts(list(zip(greedy, key)))
    os_, es = wrong_counts([(x, k) for r, k in zip(L, key) for x in r])
    from scipy.stats import chisquare
    p_g = chisquare([og[x] for x in "ABCD"], [eg[x] for x in "ABCD"]).pvalue
    for x in "ABCD":
        add("position", f"greedy_wrong_chose_{x}", og[x], n=sum(og.values()), note=f"expected {eg[x]:.1f}")
        add("position", f"sampled_wrong_chose_{x}", os_[x], n=sum(os_.values()), note=f"expected {es[x]:.1f}")
        s = key == x
        add("position", f"accuracy_greedy_key_{x}", g[s].mean(), n=int(s.sum()))
        add("position", f"accuracy_vote_key_{x}", v[s].mean(), n=int(s.sum()))
    add("position", "greedy_wrong_letters_chi2_p", p_g, n=sum(og.values()), note="item level, df 3")

    # 6. Majority of k averaged over random run orders (tie rule applied in the permuted order).
    rng = np.random.default_rng(seed)
    curve = np.zeros((N_ORDERS, 10))
    for o in range(N_ORDERS):
        perm = rng.permutation(10)
        Lp = L[:, perm]
        for k in range(1, 11):
            curve[o, k - 1] = np.mean([majority_vote(list(r[:k])).letter == kk for r, kk in zip(Lp, key)])
    for k in range(1, 11):
        add("majority_of_k_orders", f"k{k}", curve[:, k - 1].mean(), np.percentile(curve[:, k - 1], 2.5),
            np.percentile(curve[:, k - 1], 97.5), len(t), f"{N_ORDERS} random run orders; band = 2.5-97.5 percentile over orders")

    # 7. Probe hits by option length (words) and by origin; aggregate counts only.
    probe = read_jsonl(raw / "probe.jsonl")
    if probe:
        items = {it.id: it for it in select_items(load_medqa(a.raw_dir), [r["id"] for r in probe])}
        for r in probe:
            target = probe_messages(items[r["id"]])[1]
            r["_hit"], r["_short"] = exact_match(r["text"], target), len(target.split()) <= 2
        for name, sel in (("short_1_2_words", [r for r in probe if r["_short"]]),
                          ("long_3plus_words", [r for r in probe if not r["_short"]])):
            add("probe", f"hits_{name}", sum(r["_hit"] for r in sel), n=len(sel))
        for o in ("train", "dev", "test"):
            sel = [r for r in probe if r["id"].startswith(o + "-")]
            add("probe", f"hits_origin_{o}", sum(r["_hit"] for r in sel), n=len(sel))

    # 8. b-items: greedy-letter frequency among sampled runs without the unparseable-greedy items.
    b_items = (v == 1) & (g == 0)
    parsed = np.array([x is not None for x in greedy])
    cnt = np.array([sum(x == gr for x in r) if gr is not None else 0 for r, gr in zip(L, greedy)])
    add("error_analysis", "b_items_greedy_letter_runs_parsed_only", cnt[b_items & parsed].mean(), n=int((b_items & parsed).sum()))

    out = res / "posthoc_review.csv"
    pd.DataFrame(rows).to_csv(out, index=False)

    # Figure 2b: majority of k, first-k curve (frozen) and order-averaged curve (post hoc), greedy as a line.
    first_k = [np.mean([majority_vote(list(r[:k])).letter == kk for r, kk in zip(L, key)]) for k in range(1, 11)]
    ks = np.arange(1, 11)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    a1.fill_between(ks, 100 * np.percentile(curve, 2.5, axis=0), 100 * np.percentile(curve, 97.5, axis=0),
                    color=ORANGE, alpha=0.18, linewidth=0)
    a1.plot(ks, 100 * curve.mean(axis=0), color=ORANGE, linewidth=1.5, marker="s", markersize=4,
            label=f"mean over {N_ORDERS} run orders (post hoc)")
    a1.plot(ks, 100 * np.array(first_k), color=BLUE, linewidth=1.5, marker="o", markersize=4, label="first k runs (protocol)")
    a1.axhline(100 * g.mean(), color=INK2, linewidth=1.2, linestyle="--", label="greedy (1 run, T = 0)")
    a1.set_xlabel("sampled runs voted on (k)", fontsize=8)
    a1.set_ylabel("accuracy (%)", fontsize=8)
    a1.set_xticks(ks)
    fig.legend(*a1.get_legend_handles_labels(), frameon=False, fontsize=7, labelcolor=INK, loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, 0.97))
    tok = t["sampled_tokens"].mean() / 10
    a2.plot(ks, ks * tok, color=BLUE, linewidth=1.5, marker="o", markersize=4)
    a2.axhline(t["greedy_tokens"].mean(), color=INK2, linewidth=1.2, linestyle="--")
    a2.annotate("greedy", (10, t["greedy_tokens"].mean()), textcoords="offset points", xytext=(0, 4), ha="right", fontsize=7, color=INK2)
    a2.set_xlabel("sampled runs voted on (k)", fontsize=8)
    a2.set_ylabel("output tokens per item", fontsize=8)
    a2.set_xticks(ks)
    for ax in (a1, a2):
        _style(ax)
    fig.tight_layout()
    figs = _save(fig, res / "figures", "fig2b_majority_of_k_orders")
    print(f"{len(rows)} rows -> {out}; figure: {', '.join(p.name for p in figs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
