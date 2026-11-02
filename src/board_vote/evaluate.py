"""Module 4: turn saved outputs into the protocol's metrics (RQ1-RQ3). CPU only. Written and tested
before any test-set result exists.

Input:  item info (id -> key, subgroup code M/O, objective-subset flag) and the saved records of the
        greedy, sampled and greedy_rerun arms (module 3 format).
Output: an item-level table (letters and flags only, no text) and a long table of metrics
        (metric, subset, value, ci_low, ci_high, n, note) for results.csv.
Checks first: every item has every expected output and no unknown item appears (else ValueError).
"""
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from .parse import AnswerParser, majority_of_first_k, majority_vote

CATEGORIES = ["A", "B", "C", "D", None]          # None = unparseable, a category for Fleiss' kappa
K_CURVE = [1, 3, 5, 7, 9, 10]


# ---------- statistics (each tested against a known answer) ----------

def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value. b = vote right & greedy wrong, c = greedy right & vote wrong."""
    return 1.0 if b + c == 0 else binomtest(b, b + c, 0.5).pvalue


def paired_bootstrap_ci(x: np.ndarray, y: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    """95% percentile CI of mean(x) - mean(y), resampling items (paired)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    diffs = x[idx].mean(axis=1) - y[idx].mean(axis=1)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def bootstrap_ci(values: np.ndarray, stat, n_boot: int, seed: int) -> tuple[float, float]:
    """95% percentile CI of stat(values) over item resamples (values: one row per item)."""
    rng = np.random.default_rng(seed)
    stats = [stat(values[rng.integers(0, len(values), len(values))]) for _ in range(n_boot)]
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def fleiss_kappa(counts: np.ndarray) -> float:
    """Fleiss' kappa from an items x categories matrix of rating counts (same number of raters per item)."""
    counts = np.asarray(counts, dtype=float)
    n = counts.sum(axis=1)[0]
    p_j = counts.sum(axis=0) / counts.sum()
    p_i = ((counts ** 2).sum(axis=1) - n) / (n * (n - 1))
    p_bar, p_e = p_i.mean(), (p_j ** 2).sum()
    return float("nan") if p_e == 1 else float((p_bar - p_e) / (1 - p_e))


def auroc(scores, labels) -> float:
    """Probability that a correct item has a higher score than an incorrect one (ties count half)."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


# ---------- item-level table ----------

def _group(records: list[dict]) -> dict[str, list[dict]]:
    g = defaultdict(list)
    for r in records:
        g[r["id"]].append(r)
    return {i: sorted(rs, key=lambda r: r["run"]) for i, rs in g.items()}


def build_item_table(items: pd.DataFrame, records: dict[str, list[dict]], parser: AnswerParser,
                     n_sampled: int) -> pd.DataFrame:
    """items: columns id, key, code, objective_subset. records: arm -> list of saved records."""
    ids = list(items["id"])
    g, s = _group(records["greedy"]), _group(records["sampled"])
    for arm, grp, n in (("greedy", g, 1), ("sampled", s, n_sampled)):
        extra = set(grp) - set(ids)
        missing = [i for i in ids if i not in grp or [r["run"] for r in grp[i]] != list(range(n))]
        if extra or missing:
            raise ValueError(f"{arm}: {len(missing)} items missing/incomplete, {len(extra)} unknown items")
    rows = []
    for it in items.itertuples(index=False):
        gl = parser(g[it.id][0]["text"])
        letters = [parser(r["text"]) for r in s[it.id]]
        v = majority_vote(letters)
        rows.append({
            "id": it.id, "key": it.key, "code": it.code, "objective_subset": bool(it.objective_subset),
            "greedy": gl, "greedy_correct": gl == it.key,
            "greedy_truncated": g[it.id][0]["finish"] == "length",
            "vote": v.letter, "vote_correct": v.letter == it.key, "vote_count": v.count,
            "modal_share": v.modal_share, "vote_tie": v.tie,
            **{f"run{j}": letters[j] for j in range(n_sampled)},
            **{f"run{j}_correct": letters[j] == it.key for j in range(n_sampled)},
            **{f"vote_k{k}_correct": majority_of_first_k(letters, k).letter == it.key for k in K_CURVE},
            "sampled_unparseable": sum(x is None for x in letters),
            "sampled_truncated": sum(r["finish"] == "length" for r in s[it.id]),
            "greedy_tokens": g[it.id][0]["new_tokens"],
            "sampled_tokens": sum(r["new_tokens"] for r in s[it.id]),
            "greedy_gpu_s": g[it.id][0]["batch_seconds"] / g[it.id][0]["batch_items"],
            "sampled_gpu_s": s[it.id][0]["batch_seconds"] / s[it.id][0]["batch_items"],
        })
    return pd.DataFrame(rows)


# ---------- metrics ----------

def _row(metric, subset, value, ci=(np.nan, np.nan), n=None, note=""):
    return {"metric": metric, "subset": subset, "value": value, "ci_low": ci[0], "ci_high": ci[1],
            "n": n, "note": note}


def primary(t: pd.DataFrame, subset: str, n_boot: int, seed: int, lenient: bool = False) -> list[dict]:
    if lenient:
        t = t[t["greedy"].notna() & t["vote"].notna()]
    v, g = t["vote_correct"].to_numpy(float), t["greedy_correct"].to_numpy(float)
    b = int(((t["vote_correct"]) & (~t["greedy_correct"])).sum())
    c = int(((~t["vote_correct"]) & (t["greedy_correct"])).sum())
    diff, ci, p = v.mean() - g.mean(), paired_bootstrap_ci(v, g, n_boot, seed), mcnemar_exact(b, c)
    tag = subset + (" (lenient)" if lenient else "")
    decision = ("supported" if p < 0.05 and ci[0] > 0 else "not supported") + \
               ("; meaningful (>= 3 points)" if diff >= 0.03 and p < 0.05 and ci[0] > 0 else "")
    return [
        _row("accuracy_difference_majority10_minus_greedy", tag, diff, ci, len(t),
             f"McNemar exact p={p:.4g}; discordant b={b} (vote only right), c={c} (greedy only right); H1 {decision}"),
        _row("accuracy_greedy", tag, g.mean(), bootstrap_ci(g, np.mean, n_boot, seed + 1), len(t)),
        _row("accuracy_majority10", tag, v.mean(), bootstrap_ci(v, np.mean, n_boot, seed + 2), len(t)),
    ]


def evaluate(items: pd.DataFrame, records: dict[str, list[dict]], cfg: dict, n_boot: int = 10000) -> tuple[pd.DataFrame, pd.DataFrame]:
    parser = AnswerParser(cfg["parsing"])
    n_s = cfg["decoding"]["sampled"]["n"]
    seed = cfg["model"]["engine_args"]["seed"]
    t = build_item_table(items, records, parser, n_s)
    rows = []

    # RQ1 (confirmatory) and sensitivity analyses.
    rows += primary(t, "all", n_boot, seed)
    rows += primary(t, "all", n_boot, seed, lenient=True)
    rows += primary(t[t["objective_subset"]], "objective_subset", n_boot, seed)
    for code in ("M", "O"):
        rows += primary(t[t["code"] == code], f"subgroup_{code}", n_boot, seed)

    run_acc = np.array([t[f"run{j}_correct"].mean() for j in range(n_s)])
    rows.append(_row("accuracy_single_sample_mean", "all", run_acc.mean(), (np.nan, np.nan), len(t),
                     f"SD over the {n_s} runs = {run_acc.std(ddof=1):.4f}; min {run_acc.min():.4f}, max {run_acc.max():.4f}"))
    for name, col in (("chance_corrected_accuracy_greedy", "greedy_correct"),
                      ("chance_corrected_accuracy_majority10", "vote_correct")):
        rows.append(_row(name, "all", (t[col].mean() - 0.25) / 0.75, n=len(t), note="(acc - 0.25) / 0.75"))
    for k in K_CURVE:
        rows.append(_row(f"accuracy_majority_of_first_{k}", "all", t[f"vote_k{k}_correct"].mean(), n=len(t)))

    # RQ2 consistency.
    # Unparseable runs are None; pandas 3 stores them as NaN in the table, so map them back to None.
    letters = [[None if pd.isna(x) else x for x in r] for r in t[[f"run{j}" for j in range(n_s)]].to_numpy(object)]
    unanimous = np.array([len(set(r)) == 1 and r[0] is not None for r in letters], float)
    rows.append(_row("unanimity_rate", "all", unanimous.mean(), bootstrap_ci(unanimous, np.mean, n_boot, seed + 3),
                     len(t), "share of items with 10/10 identical parsed letters"))
    ms = t["modal_share"].to_numpy(float)
    rows.append(_row("mean_modal_share", "all", ms.mean(), bootstrap_ci(ms, np.mean, n_boot, seed + 4), len(t)))
    counts = np.array([[sum(x == c for x in r) for c in CATEGORIES] for r in letters])
    rows.append(_row("fleiss_kappa_across_runs", "all", fleiss_kappa(counts),
                     bootstrap_ci(counts, fleiss_kappa, min(n_boot, 2000), seed + 5), len(t),
                     "categories A-D + unparseable; bootstrap 2000 resamples"))
    flips = np.array([np.mean([a != b for a, b in combinations(r, 2)]) for r in letters])
    rows.append(_row("pairwise_flip_rate", "all", flips.mean(), bootstrap_ci(flips, np.mean, n_boot, seed + 6), len(t),
                     "share of the 45 run pairs with different answers, averaged over items"))

    # RQ3 agreement as a reliability flag.
    lab = t["vote_correct"].to_numpy(bool)
    both = np.column_stack([ms, lab])
    rows.append(_row("auroc_modal_share_vs_majority_correct", "all", auroc(ms, lab),
                     bootstrap_ci(both, lambda a: auroc(a[:, 0], a[:, 1].astype(bool)), min(n_boot, 2000), seed + 7),
                     len(t), "H3 supported if CI lower bound > 0.5; bootstrap 2000 resamples"))
    hi = t["vote_count"] >= 9
    rows.append(_row("accuracy_agreement_ge9", "all", t.loc[hi, "vote_correct"].mean(), n=int(hi.sum())))
    rows.append(_row("accuracy_agreement_lt9", "all", t.loc[~hi, "vote_correct"].mean(), n=int((~hi).sum())))
    for x in range(1, n_s + 1):
        sel = t["vote_count"] >= x
        rows.append(_row(f"coverage_at_agreement_ge{x}", "all", sel.mean(), n=len(t),
                         note=f"accuracy among covered = {t.loc[sel, 'vote_correct'].mean():.4f}" if sel.any() else ""))

    # Output quality and cost.
    rows.append(_row("unparseable_rate_greedy", "all", t["greedy"].isna().mean(), n=len(t)))
    rows.append(_row("unparseable_rate_sampled", "all", t["sampled_unparseable"].sum() / (n_s * len(t)), n=n_s * len(t)))
    rows.append(_row("truncation_rate_greedy", "all", t["greedy_truncated"].mean(), n=len(t)))
    rows.append(_row("truncation_rate_sampled", "all", t["sampled_truncated"].sum() / (n_s * len(t)), n=n_s * len(t)))
    rows.append(_row("vote_ties", "all", int(t["vote_tie"].sum()), n=len(t), note="decided by the earliest-run rule"))
    for arm in ("greedy", "sampled"):
        rows.append(_row(f"output_tokens_total_{arm}", "all", int(t[f"{arm}_tokens"].sum()), n=len(t)))
        rows.append(_row(f"gpu_seconds_{arm}", "all", float(t[f"{arm}_gpu_s"].sum()), n=len(t),
                         note="engine wall time for the arm, summed over batches"))

    # Determinism check (greedy rerun subset).
    if records.get("greedy_rerun"):
        g1 = {r["id"]: r for r in records["greedy"]}
        rr = records["greedy_rerun"]
        same_text = np.mean([g1[r["id"]]["text"] == r["text"] for r in rr])
        same_letter = np.mean([parser(g1[r["id"]]["text"]) == parser(r["text"]) for r in rr])
        rows.append(_row("greedy_rerun_identical_letter", "rerun_subset", same_letter, n=len(rr)))
        rows.append(_row("greedy_rerun_identical_text", "rerun_subset", same_text, n=len(rr)))

    return t, pd.DataFrame(rows)
