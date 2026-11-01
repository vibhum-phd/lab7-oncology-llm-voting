"""Prompt 06 data checks on the screened MedQA oncology items (CPU only).

Checks: exact and near-duplicate vignettes (within the eligible set and against all of MedQA),
items that refer to an image or figure the model cannot see, label balance, lengths.
Usage: python -I -X utf8 scripts/validate_items.py data/raw/medqa_4opt_hf data/screening [CODES_CSV]
CODES_CSV (default ai_codes.csv) is read from the screening folder; flags go to validation_flags.csv.
"""
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd

# Visual material the text-only item does not contain. "Results are shown below:" followed by the
# values in the text is not counted.
IMAGE_RE = re.compile(
    r"(see (the )?(image|figure|photograph|picture)|shown in the (image|figure|photograph|picture|exhibit)|"
    r"(image|photograph|figure|picture|radiograph|x-ray|scan|mri|ecg|electrocardiogram|smear|"
    r"micrograph|photomicrograph|specimen|biopsy|lesion|rash|findings?)s? (is|are) shown(?! below:)|"
    r"\bexhibit\b|image below|figure below|(the|this) (image|photograph|figure)\b)",
    re.IGNORECASE,
)


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", text.lower())).strip()


def shingles(text: str, k: int = 5) -> set:
    w = norm(text).split()
    return {" ".join(w[i:i + k]) for i in range(max(1, len(w) - k + 1))}


def main(raw: str, screening: str, codes_file: str = "ai_codes.csv") -> None:
    allq = pd.concat([pd.read_parquet(Path(raw) / f"{s}.parquet") for s in ("train", "validation", "test")])
    codes = pd.read_csv(Path(screening) / codes_file)
    print(f"codes file: {codes_file}")
    elig = codes[codes["code"].isin(["M", "O"])].merge(allq, on="id", validate="one_to_one")
    print(f"eligible (AI codes M/O): {len(elig)}  {elig['code'].value_counts().to_dict()}")

    # Exact duplicates on normalised vignette text, across all of MedQA.
    allq["n"] = allq["sent1"].map(norm)
    dup_groups = allq[allq.duplicated("n", keep=False)].groupby("n")["id"].apply(list)
    elig_ids = set(elig["id"])
    exact = [g for g in dup_groups if elig_ids & set(g)]
    print(f"exact-duplicate vignette groups in all MedQA: {len(dup_groups)}; touching eligible items: {len(exact)}")
    for g in exact[:10]:
        print("   ", g)

    # Near duplicates within the eligible set: Jaccard of 5-word shingles >= 0.6.
    sh = {r["id"]: shingles(r["sent1"]) for _, r in elig.iterrows()}
    index = defaultdict(set)
    for i, s in sh.items():
        for g in s:
            index[g].add(i)
    cand = set()
    for ids in index.values():
        if 1 < len(ids) < 30:
            cand.update(combinations(sorted(ids), 2))
    near = []
    for a, b in cand:
        j = len(sh[a] & sh[b]) / len(sh[a] | sh[b])
        if j >= 0.6:
            near.append((round(j, 2), a, b))
    near.sort(reverse=True)
    print(f"near-duplicate pairs within eligible set (Jaccard >= 0.6): {len(near)}")
    for row in near[:15]:
        print("   ", row)

    # Image / figure references.
    elig["image_ref"] = elig["sent1"].str.contains(IMAGE_RE)
    print(f"eligible items referring to an image/figure/exhibit: {int(elig['image_ref'].sum())}")
    for _, r in elig[elig["image_ref"]].head(6).iterrows():
        m = IMAGE_RE.search(r["sent1"])
        s = max(0, m.start() - 60)
        print(f"    {r['id']}: ...{r['sent1'][s:m.end() + 40]}...".replace("\n", " "))

    # Label balance and lengths.
    print("label balance (0-3):", elig["label"].value_counts().sort_index().to_dict())
    words = elig["sent1"].str.split().str.len()
    print(f"vignette words: median {int(words.median())}, p90 {int(words.quantile(0.9))}, max {int(words.max())}")
    # Options that themselves look like letters or combined answers can confuse letter parsing.
    endings = ["ending0", "ending1", "ending2", "ending3"]
    odd = elig[elig[endings].apply(lambda r: any(re.fullmatch(r"\(?[A-Ea-e]\)?", x.strip()) for x in r), axis=1)]
    print(f"items whose options are bare letters (parse risk): {len(odd)}  {list(odd['id'])[:10]}")

    flags = elig[["id", "code", "image_ref"]].copy()
    near_ids = {x for _, a, b in near for x in (a, b)}
    exact_ids = {x for g in exact for x in g}
    flags["near_dup"] = flags["id"].isin(near_ids)
    flags["exact_dup"] = flags["id"].isin(exact_ids)
    flags["bare_letter_options"] = flags["id"].isin(set(odd["id"]))
    flags.to_csv(Path(screening) / "validation_flags.csv", index=False)
    print("flags written to validation_flags.csv")


if __name__ == "__main__":
    main(*sys.argv[1:4])
