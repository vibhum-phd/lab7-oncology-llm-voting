"""Draw Sir's blind 20% re-screening sample and list the AI's 'unsure' items with full text.

Usage: python -I -X utf8 scripts/make_check_sample.py data/raw/medqa_4opt_hf data/screening
Writes sir_check_sample.csv (no AI codes shown) and sir_unsure_items.csv into the screening folder.
"""
import sys
from pathlib import Path

import pandas as pd

SEED = 20261101
FRACTION = 0.20


def main(raw: str, screening: str) -> None:
    items = pd.concat([pd.read_parquet(Path(raw) / f"{s}.parquet") for s in ("train", "validation", "test")])
    codes = pd.read_csv(Path(screening) / "ai_codes.csv")
    df = codes.merge(items, on="id", how="left", validate="one_to_one")
    df["correct_answer"] = df.apply(lambda r: r[f"ending{int(r['label'])}"], axis=1)
    cols = ["id", "sent1", "ending0", "ending1", "ending2", "ending3", "correct_answer"]

    sample = df.sample(frac=FRACTION, random_state=SEED).sort_values("id")
    out = sample[cols].rename(columns={"sent1": "vignette_and_question"})
    out["sir_code (M/O/X)"] = ""
    out.to_csv(Path(screening) / "sir_check_sample.csv", index=False, encoding="utf-8-sig")

    unsure = df[df["code"] == "U"][cols].rename(columns={"sent1": "vignette_and_question"})
    unsure["sir_code (M/O/X)"] = ""
    unsure.to_csv(Path(screening) / "sir_unsure_items.csv", index=False, encoding="utf-8-sig")

    print(f"check sample: {len(sample)} items (seed {SEED}); AI codes in sample: "
          f"{sample['code'].value_counts().to_dict()}")
    print(f"unsure items: {len(unsure)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
