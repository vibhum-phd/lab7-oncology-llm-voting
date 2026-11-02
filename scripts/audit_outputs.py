"""Audit saved outputs before any metric is trusted (Prompt 13). CPU only; prints no item text.

Usage (from repo/):
  python -X utf8 scripts/audit_outputs.py --split test --raw results/raw/test --out results \
      --env results/env/kaggle_03_test_v1 results/env/kaggle_04_extras_v1
Checks: partitions and checksums, ID/label alignment, missing values, that every arm covers the same items with the
frozen run counts, seeds, token limit, engine and model revision, unparseable and truncated answers, metric direction,
and leakage (answer key into prompts, items across splits, protocol changed after the runs).
Writes <out>/audit_report.json; exit code 1 if any check fails.
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from board_vote.config import load_config  # noqa: E402
from board_vote.data import LETTERS, Item, load_medqa, read_ids, select_items  # noqa: E402
from board_vote.generate import item_seed, runs_per_item  # noqa: E402
from board_vote.parse import AnswerParser  # noqa: E402
from board_vote.probe import probe_ids  # noqa: E402
from board_vote.prompts import build_messages  # noqa: E402

FIELDS = ["id", "arm", "run", "seed", "text", "new_tokens", "finish", "batch_seconds", "batch_items", "engine",
          "model_revision"]
RERUN_N, PROBE_MAX_TOKENS = 50, 64           # as in scripts/run_generation.py
checks = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    checks.append({"check": name, "ok": bool(ok), "detail": str(detail)})


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["debug", "test"], required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--env", nargs="*", default=[], help="Kaggle output folders (log + pip_freeze.txt) of the runs")
    ap.add_argument("--raw-dir", default=str(ROOT / "data" / "raw" / "medqa_4opt_hf"))
    a = ap.parse_args()

    cfg, cfg2 = load_config(), load_config(ROOT / "configs" / "protocol_v2.yaml")
    dec, base = cfg["decoding"], cfg["model"]["engine_args"]["seed"]
    splits = ROOT / "data" / "splits"
    manifest = json.loads((splits / "SPLIT_MANIFEST.json").read_text(encoding="utf-8"))
    final = pd.read_csv(splits / "items_final.csv")

    # 1. Partitions: checksums, no overlap, split labels agree with the ID files.
    ids_by = {s: sorted(read_ids(splits / f"{s}_ids.txt")) for s in ("debug", "test")}
    for s, ids in ids_by.items():
        sha = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
        check(f"{s} IDs match the manifest checksum and count", sha == manifest["splits"][s]["sha256"]
              and len(ids) == manifest["splits"][s]["n"], f"n = {len(ids)}")
    check("debug and test IDs do not overlap", not set(ids_by["debug"]) & set(ids_by["test"]))
    check("items_final split column agrees with the ID files",
          all(sorted(final.loc[final["split"] == s, "id"]) == ids for s, ids in ids_by.items()))
    ids = ids_by[a.split]
    index = {i: k for k, i in enumerate(ids)}

    # 2. ID/label alignment and missing values in the source items.
    medqa = load_medqa(a.raw_dir)
    items = select_items(medqa, ids)
    keys = {it.id: it.key for it in items}
    check("every item has a key A-D and four non-empty options",
          all(it.key in LETTERS and all(o for o in it.options) and it.question for it in items))
    dist = pd.Series(list(keys.values())).value_counts().sort_index().to_dict()
    check("answer keys are not concentrated on one letter (max share < 0.4)", max(dist.values()) / len(ids) < 0.4, dist)

    # 3. Leakage: same question text in two splits, answer key reaching the prompt.
    norm = lambda q: re.sub(r"\s+", " ", q.lower()).strip()
    qhash = {s: {hashlib.sha1(norm(str(medqa.loc[i, "sent1"])).encode()).hexdigest() for i in v} for s, v in ids_by.items()}
    check("no question text shared by debug and test", not qhash["debug"] & qhash["test"])
    check("no duplicate question text inside the split", len(qhash[a.split]) == len(ids), f"{len(qhash[a.split])} unique")
    swap = lambda it: Item(it.id, it.question, it.options, LETTERS[(LETTERS.index(it.key) + 1) % 4])
    check("prompts do not depend on the answer key (key changed -> identical prompt)",
          all(build_messages(it, cfg["prompt"]) == build_messages(swap(it), cfg["prompt"]) for it in items))

    # 4. Every arm: same items, frozen run counts, seeds, token limit, engine and revision, no missing values.
    parse = AnswerParser(cfg["parsing"])
    expected = {"greedy": ids, "sampled": ids, "greedy_rerun": ids[:RERUN_N], "greedy_rerun_full": ids,
                "sampled_t03": ids, "sampled_t10": ids, "probe": probe_ids(ids, base)}
    raw, arms, letters, summary = Path(a.raw), {}, {}, {}
    for arm, want in expected.items():
        recs = read_jsonl(raw / f"{arm}.jsonl")
        if not recs:
            print(f"[SKIP] {arm}: no output file")
            continue
        arms[arm] = recs
        n = runs_per_item(arm, dec)
        max_tok = PROBE_MAX_TOKENS if arm == "probe" else dec["max_tokens"]
        pairs = [(r["id"], r["run"]) for r in recs]
        check(f"{arm}: covers exactly the expected {len(want)} items x {n} run(s), no duplicates",
              sorted(pairs) == sorted((i, j) for i in want for j in range(n)), f"{len(recs)} records")
        check(f"{arm}: no missing fields or empty values",
              all(all(f in r and r[f] is not None for f in FIELDS) and r["finish"] in ("stop", "length") for r in recs))
        check(f"{arm}: arm label, seed rule (base + item index), engine and model revision",
              all(r["arm"] == arm and r["seed"] == item_seed(base, index[r["id"]])
                  and r["model_revision"] == cfg["model"]["revision"] for r in recs)
              and len({r["engine"] for r in recs}) == 1, recs[0]["engine"])
        trunc = sum(r["finish"] == "length" for r in recs)
        check(f"{arm}: token limit {max_tok} respected; 'length' only at the limit",
              all(r["new_tokens"] <= max_tok and (r["finish"] != "length" or r["new_tokens"] == max_tok) for r in recs),
              f"{trunc} truncated")
        if arm != "probe":
            got = [parse(r["text"]) for r in recs]
            letters[arm] = {(r["id"], r["run"]): x for r, x in zip(recs, got)}
            unp = [r for r, x in zip(recs, got) if x is None]
            summary[arm] = {"records": len(recs), "unparseable": len(unp),
                            "unparseable_truncated": sum(r["finish"] == "length" for r in unp),
                            "unparseable_not_truncated": sum(r["finish"] == "stop" for r in unp), "truncated": trunc}
    check("all arms come from the same engine version", len({r[0]["engine"] for r in arms.values()}) == 1)
    print("unparseable / truncated per arm:", json.dumps(summary))

    # 5. Metric direction: accuracy is "letter == key"; it must fall to about chance if keys are shuffled.
    g = np.array([letters["greedy"][(i, 0)] == keys[i] for i in ids], float)
    rng = np.random.default_rng(base)
    k_arr, pred = np.array([keys[i] for i in ids]), np.array([letters["greedy"][(i, 0)] for i in ids], object)
    shuffled = np.mean([np.mean(pred == rng.permutation(k_arr)) for _ in range(200)])
    check("greedy accuracy far above the shuffled-key accuracy (alignment is real)",
          g.mean() - shuffled > 0.25 and shuffled < 0.35, f"true {g.mean():.4f}, shuffled keys {shuffled:.4f}")
    res = Path(a.out) / "results.csv"
    if res.exists():
        r = pd.read_csv(res)
        get = lambda m: r.loc[(r.metric == m) & (r.subset == "all"), "value"].iloc[0]
        check("results.csv: greedy accuracy recomputed independently", abs(get("accuracy_greedy") - g.mean()) < 1e-12)
        check("results.csv: difference = majority - greedy (positive favours voting)",
              abs(get("accuracy_difference_majority10_minus_greedy")
                  - (get("accuracy_majority10") - get("accuracy_greedy"))) < 1e-12)

    # 6. Runs were made with the frozen protocol, and the software did not change between Kaggle sessions.
    frozen = (ROOT / "configs" / "protocol_v1.yaml").read_text(encoding="utf-8").replace("\r\n", "\n")
    check("protocol v1 is FROZEN and v2 keeps it unchanged", cfg["status"] == "FROZEN"
          and all(cfg[k] == cfg2[k] for k in cfg if k != "version"))
    freezes = []
    for env in map(Path, a.env):
        log = next(env.glob("*.log"), None)
        text = "".join(e.get("data", "") for e in json.loads(log.read_text(encoding="utf-8"))) if log else ""
        commit = re.search(r"commit ([0-9a-f]{40})", text)
        at_run = git("show", f"{commit.group(1)}:configs/protocol_v1.yaml") if commit else ""
        check(f"{env.name}: run used commit {commit.group(1)[:7] if commit else '?'} whose protocol_v1.yaml equals today's",
              bool(commit) and at_run == frozen)
        freezes.append((env / "pip_freeze.txt").read_text(encoding="utf-8") if (env / "pip_freeze.txt").exists() else None)
    if len(freezes) > 1:
        check("pip freeze identical across the Kaggle sessions", None not in freezes and len(set(freezes)) == 1)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"split": a.split, "n_items": len(ids), "key_distribution": dist, "per_arm": summary,
              "checks_passed": sum(c["ok"] for c in checks), "checks_total": len(checks), "checks": checks}
    (out / "audit_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"{report['checks_passed']}/{report['checks_total']} audit checks passed -> {out / 'audit_report.json'}")
    return 0 if all(c["ok"] for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
