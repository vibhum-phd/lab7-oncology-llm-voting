"""Error and agreement patterns behind the RQ1-RQ3 results (Prompt 14). CPU only; writes no item text.

Usage (from repo/):
  python -X utf8 scripts/error_patterns.py --split test --raw results/raw/test --results results
Reads <results>/item_level.csv and the raw outputs; writes <results>/error_patterns.csv (descriptive, mostly
post hoc; each row says whether it was pre-specified) and <results>/error_example_ids.csv (IDs drawn at random,
seed fixed, for reading real examples; IDs only).
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
from board_vote.evaluate import mcnemar_exact, paired_bootstrap_ci  # noqa: E402
from board_vote.parse import AnswerParser  # noqa: E402

N_EXAMPLES = 2


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["debug", "test"], required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    cfg = load_config()
    parse, seed, raw, res = AnswerParser(cfg["parsing"]), cfg["model"]["engine_args"]["seed"], Path(a.raw), Path(a.results)
    t = pd.read_csv(res / "item_level.csv", keep_default_na=False, na_values=[""])
    runs = [f"run{j}" for j in range(10)]
    L = t[runs].astype(object).where(t[runs].notna(), None).to_numpy(object)
    key, greedy = t["key"].to_numpy(object), t["greedy"].astype(object).where(t["greedy"].notna(), None).to_numpy(object)
    gc, vc = t["greedy_correct"].astype(bool).to_numpy(), t["vote_correct"].astype(bool).to_numpy()
    rc = t[[f"{r}_correct" for r in runs]].astype(bool).to_numpy()
    rows = []
    add = lambda group, metric, value, n, pre=False, note="": rows.append(
        {"group": group, "metric": metric, "value": value, "n": n, "pre_specified": pre, "note": note})

    # 1. Discordant items (pre-specified: error analysis of discordant items).
    b, c = vc & ~gc, gc & ~vc
    same_as_greedy = np.array([sum(x == g for x in r) if g is not None else 0 for r, g in zip(L, greedy)])
    key_count = np.array([sum(x == k for x in r) for r, k in zip(L, key)])
    for name, sel in (("b: vote right, greedy wrong", b), ("c: greedy right, vote wrong", c)):
        add(name, "items", int(sel.sum()), len(t), True)
        add(name, "mean sampled runs giving greedy's letter (of 10)", same_as_greedy[sel].mean(), int(sel.sum()), True)
        add(name, "mean sampled runs giving the key (of 10)", key_count[sel].mean(), int(sel.sum()), True)
        add(name, "median vote count (of 10)", float(np.median(t.loc[sel, "vote_count"])), int(sel.sum()), True)
        add(name, "votes decided by the tie rule", int(t.loc[sel, "vote_tie"].astype(bool).sum()), int(sel.sum()), True)
        add(name, "greedy unparseable", int(pd.isna(t.loc[sel, "greedy"]).sum()), int(sel.sum()), True)
        add(name, "management (M) items", int((t.loc[sel, "code"] == "M").sum()), int(sel.sum()), True)
    add("all items", "greedy letter equals the vote", float(np.mean([g == v for g, v in zip(greedy, t["vote"])])), len(t),
        note="unparseable greedy counts as different")

    # 2. Items wrong in all 11 generations (pre-specified).
    w = ~gc & ~rc.any(axis=1)
    distinct = np.array([len({x for x in list(r) + [g] if x is not None}) for r, g in zip(L, greedy)])
    one_wrong = w & (distinct == 1) & np.array([None not in list(r) + [g] for r, g in zip(L, greedy)])
    add("wrong in all 11", "items", int(w.sum()), len(t), True)
    add("wrong in all 11", "same wrong letter in all 11 generations", int(one_wrong.sum()), int(w.sum()), True)
    add("wrong in all 11", "two or more different wrong letters", int((w & (distinct >= 2)).sum()), int(w.sum()), True)
    add("wrong in all 11", "management (M) items", int((t.loc[w, "code"] == "M").sum()), int(w.sum()), True,
        f"M share overall {np.mean(t['code'] == 'M'):.3f}")
    add("wrong in all 11", "objective-subset items", int(t.loc[w, "objective_subset"].astype(bool).sum()), int(w.sum()), True,
        f"objective share overall {t['objective_subset'].astype(bool).mean():.3f}")

    # 3. Confident errors: unanimous (10/10) but wrong (RQ3 context; post hoc).
    unanimous = np.array([len(set(r)) == 1 and r[0] is not None for r in L])
    add("10/10 unanimous", "items", int(unanimous.sum()), len(t))
    add("10/10 unanimous", "of which wrong", int((unanimous & ~vc).sum()), int(unanimous.sum()))

    # 4. Reasoning length and truncation, correct vs wrong sampled runs (post hoc).
    recs = read_jsonl(raw / "sampled.jsonl")
    keyd = dict(zip(t["id"], key))
    ok = np.array([parse(r["text"]) == keyd[r["id"]] for r in recs])
    tok = np.array([r["new_tokens"] for r in recs])
    trunc = np.array([r["finish"] == "length" for r in recs])
    for name, sel in (("sampled runs, correct", ok), ("sampled runs, wrong or unparseable", ~ok)):
        add(name, "runs", int(sel.sum()), len(recs))
        add(name, "median output tokens", float(np.median(tok[sel])), int(sel.sum()))
        add(name, "truncated at 512 tokens", int(trunc[sel].sum()), int(sel.sum()))

    # 5. Letter bias: chosen letter of wrong greedy answers vs the key distribution (post hoc).
    wrong_letters = Counter(g for g, k in zip(greedy, key) if g is not None and g != k)
    keys = Counter(key)
    for letter in "ABCD":
        add("greedy wrong answers", f"chose {letter}", wrong_letters[letter], sum(wrong_letters.values()),
            note=f"key is {letter} for {keys[letter]} of {len(t)} items")

    # 6. Stable answers (>= 9 of 10 runs agree) per temperature, as in Pinhanez 2025 (post hoc, descriptive).
    for arm, temp in (("sampled_t03", 0.3), ("sampled", 0.7), ("sampled_t10", 1.0)):
        rr = read_jsonl(raw / f"{arm}.jsonl")
        if not rr:
            continue
        by = {}
        for r in rr:
            by.setdefault(r["id"], []).append(parse(r["text"]))
        top = {i: Counter(x for x in v if x is not None).most_common(1) for i, v in by.items()}
        stable = {i: bool(m) and m[0][1] >= 9 for i, m in top.items()}
        acc_st = np.mean([top[i][0][0] == keyd[i] for i in by if stable[i]])
        add(f"T = {temp}", "share of items with >= 9/10 runs agreeing", float(np.mean(list(stable.values()))), len(by),
            note=f"accuracy of the agreed answer on these items {acc_st:.4f}")

    # 7. Vote vs the full greedy rerun as comparator (post hoc robustness of RQ1).
    rerun = {r["id"]: parse(r["text"]) for r in read_jsonl(raw / "greedy_rerun_full.jsonl")}
    if rerun:
        g2 = np.array([rerun[i] == k for i, k in zip(t["id"], key)], float)
        v = vc.astype(float)
        bb, cc = int(((v == 1) & (g2 == 0)).sum()), int(((v == 0) & (g2 == 1)).sum())
        lo, hi = paired_bootstrap_ci(v, g2, 10000, seed)
        add("vote vs greedy rerun", "accuracy difference", v.mean() - g2.mean(), len(t),
            note=f"95% CI [{lo:.4f}, {hi:.4f}]; McNemar exact p={mcnemar_exact(bb, cc):.4g}; b={bb}, c={cc}")

    out = res / "error_patterns.csv"
    pd.DataFrame(rows).to_csv(out, index=False)

    # Random examples per category, for reading real outputs (IDs only).
    rng = np.random.default_rng(seed)
    cats = {"b_vote_right_greedy_wrong": b, "c_greedy_right_vote_wrong": c, "wrong_all11_same_letter": one_wrong,
            "wrong_all11_mixed_letters": w & (distinct >= 2), "unanimous_wrong": unanimous & ~vc & ~w}
    ex = [{"category": name, "id": i} for name, sel in cats.items()
          for i in sorted(rng.choice(t.loc[sel, "id"].to_numpy(), size=min(N_EXAMPLES, int(sel.sum())), replace=False))]
    pd.DataFrame(ex).to_csv(res / "error_example_ids.csv", index=False)
    print(f"{len(rows)} rows -> {out}; {len(ex)} example IDs -> {res / 'error_example_ids.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
