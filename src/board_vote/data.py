"""Module 1a: load MedQA and select the study items by ID, with validation.

Input:  the three MedQA parquet files (HF copy GBaker/MedQA-USMLE-4-options-hf) and an ID list
        (data/splits/debug_ids.txt or test_ids.txt).
Output: a list of Item records in ID-list order. Anything unexpected raises ValueError, so a bad
        item can never reach the model silently.
"""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

LETTERS = "ABCD"
SPLITS = ("train", "validation", "test")
COLUMNS = ["id", "sent1", "sent2", "ending0", "ending1", "ending2", "ending3", "label"]


@dataclass(frozen=True)
class Item:
    id: str
    question: str                         # vignette + question sentence (MedQA field `sent1`)
    options: tuple[str, str, str, str]    # in dataset order: A, B, C, D
    key: str                              # correct letter, "A".."D"


def load_medqa(raw_dir: Path | str) -> pd.DataFrame:
    """All 12,723 MedQA items from the three parquet files, indexed by id."""
    frames = [pd.read_parquet(Path(raw_dir) / f"{s}.parquet") for s in SPLITS]
    df = pd.concat(frames, ignore_index=True)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"MedQA files lack columns {missing}")
    if df["id"].duplicated().any():
        raise ValueError(f"duplicate MedQA ids: {df.loc[df['id'].duplicated(), 'id'].tolist()[:5]}")
    return df.set_index("id")


def read_ids(path: Path | str) -> list[str]:
    """IDs from a split file, one per line; blank lines and duplicates are errors."""
    ids = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()]
    if any(not i for i in ids):
        raise ValueError(f"{path}: blank line")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate ids")
    return ids


def select_items(df: pd.DataFrame, ids: list[str]) -> list[Item]:
    """Items for `ids`, in the same order. Checks every item before returning any."""
    problems = []
    absent = [i for i in ids if i not in df.index]
    if absent:
        raise ValueError(f"{len(absent)} ids not in MedQA, e.g. {absent[:5]}")
    items = []
    for i in ids:
        r = df.loc[i]
        options = tuple(str(r[f"ending{k}"]).strip() for k in range(4))
        question = str(r["sent1"]).strip()
        try:
            label = int(r["label"])
        except (TypeError, ValueError):
            label = None
        if not question:
            problems.append(f"{i}: empty question")
        if any(not o for o in options):
            problems.append(f"{i}: empty option")
        if len(set(options)) < 4:
            problems.append(f"{i}: repeated option text")
        if str(r["sent2"]).strip():
            problems.append(f"{i}: sent2 not empty (format differs from the rest)")
        if label not in range(4):
            problems.append(f"{i}: label {r['label']!r} not in 0-3")
            continue
        items.append(Item(id=i, question=question, options=options, key=LETTERS[label]))
    if problems:
        raise ValueError(f"{len(problems)} invalid items: " + "; ".join(problems[:10]))
    return items
