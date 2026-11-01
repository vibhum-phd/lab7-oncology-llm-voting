"""Build the final item list and the debug/test split (Prompt 07). Run once, at freeze time.

Eligible = code M or O in CODES_CSV, minus items flagged in validation_flags.csv (image reference or
bare-letter options). Debug set = DEBUG_N items drawn at random, stratified by M/O; test = the rest.
Also marks the objective subset (oncology term in the correct answer, rule R4) used for the
sensitivity analysis (D-023).
Usage (from repo/): python -I -X utf8 scripts/make_splits.py data/screening CODES_CSV OUT_DIR
                    [--pool data/work/candidate_pool.csv] [--seed 20261102] [--debug-n 25]
The pool file (built by count_oncology_items.py) contains item text, so it stays in data/work/ (not committed).
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256_lines(ids: list) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("screening")
    ap.add_argument("codes_csv")
    ap.add_argument("out_dir")
    ap.add_argument("--pool", default="data/work/candidate_pool.csv")
    ap.add_argument("--seed", type=int, default=20261102)
    ap.add_argument("--debug-n", type=int, default=25)
    a = ap.parse_args()

    scr = Path(a.screening)
    codes = pd.read_csv(scr / a.codes_csv)
    flags = pd.read_csv(scr / "validation_flags.csv")
    pool = pd.read_csv(a.pool)[["id", "onc_correct"]]

    elig = codes[codes["code"].isin(["M", "O"])].merge(flags.drop(columns="code"), on="id", how="left")
    excluded = elig[elig["image_ref"].fillna(False) | elig["bare_letter_options"].fillna(False)]
    final = elig.drop(excluded.index).merge(pool, on="id", how="left")
    final = final.rename(columns={"onc_correct": "objective_subset"})[["id", "code", "objective_subset"]]
    final = final.sort_values("id").reset_index(drop=True)

    # Stratified debug draw: proportional to M/O, at least one of each.
    n_m = max(1, round(a.debug_n * (final["code"] == "M").mean()))
    debug = pd.concat([
        final[final["code"] == "M"].sample(n_m, random_state=a.seed),
        final[final["code"] == "O"].sample(a.debug_n - n_m, random_state=a.seed),
    ])
    final["split"] = final["id"].isin(set(debug["id"])).map({True: "debug", False: "test"})

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    final.to_csv(out / "items_final.csv", index=False)
    lists = {}
    for split in ("debug", "test"):
        ids = sorted(final.loc[final["split"] == split, "id"])
        (out / f"{split}_ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
        lists[split] = {"n": len(ids), "sha256": sha256_lines(ids),
                        "M": int((final.loc[final["split"] == split, "code"] == "M").sum())}
    manifest = {
        "codes_file": a.codes_csv, "seed": a.seed, "debug_n": a.debug_n,
        "eligible_before_exclusions": len(elig), "excluded_image_or_bare_letter": len(excluded),
        "final_items": len(final), "objective_subset_in_test": int(final.loc[final["split"] == "test", "objective_subset"].sum()),
        "splits": lists,
    }
    (out / "SPLIT_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
