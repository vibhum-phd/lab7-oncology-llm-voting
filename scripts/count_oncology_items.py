"""Count oncology items in MedQA (4-option, HF copy) under a written keyword rule.

Exploratory count for the Prompt 05 power check. The final selection rule is frozen in Prompt 06/07.
Usage: python -I -X utf8 scripts/count_oncology_items.py data/raw/medqa_4opt_hf [POOL_OUT_DIR]
With POOL_OUT_DIR, also writes the screening candidate pool (R1 or R4) as candidate_pool.csv and a
compact text view, screening_view.txt, for item-by-item screening.
"""
import re
import sys
from pathlib import Path

import pandas as pd

ONC_TERMS = [
    r"cancer", r"carcinoma", r"adenocarcinoma", r"sarcoma", r"lymphoma", r"leuka?emia", r"myeloma",
    r"melanoma", r"tumou?r", r"neoplas\w*", r"malignan\w*", r"metasta\w*", r"oncolog\w*",
    r"chemotherap\w*", r"radiotherap\w*", r"radiation therapy", r"glioma", r"glioblastoma",
    r"\w+blastoma", r"mesothelioma", r"seminoma", r"carcinoid", r"lymphoproliferative",
    r"myelodysplastic", r"mastectomy", r"lumpectomy", r"colectomy", r"lobectomy", r"whipple",
    r"tamoxifen", r"trastuzumab", r"cisplatin", r"doxorubicin", r"imatinib", r"rituximab",
    r"pembrolizumab", r"nivolumab", r"methotrexate", r"cyclophosphamide", r"5-fluorouracil",
]
ONC_RE = re.compile(r"\b(" + "|".join(ONC_TERMS) + r")\b", re.IGNORECASE)
# Question asks for management or treatment (tumor-board-style decision), judged on the last sentence.
MGMT_RE = re.compile(
    r"(next (best )?step|management|treatment|therapy|pharmacotherapy|appropriate (intervention|"
    r"course)|most likely to (benefit|improve)|recommend)",
    re.IGNORECASE,
)


# Oncology words that only describe a relative's history, not this patient.
FAMILY_RE = re.compile(
    r"\b(family history|father|mother|brother|sister|aunt|uncle|grand(mother|father)|parent|sibling|"
    r"cousin|son|daughter)\b[^.]{0,60}$",
    re.IGNORECASE,
)


def last_sentence(text: str) -> str:
    """The question sentence: the last sentence ending in '?', else the last sentence."""
    questions = re.findall(r"[^.?!\n]*\?", text)
    if questions:
        return questions[-1].strip()
    parts = re.split(r"(?<=[.!])\s+", text.strip())
    return parts[-1] if parts else text


def patient_oncology(text: str) -> bool:
    """True if an oncology term appears that is not just part of a relative's history."""
    for m in ONC_RE.finditer(text):
        if not FAMILY_RE.search(text[max(0, m.start() - 80):m.start()]):
            return True
    return False


def term_contexts(text: str, limit: int = 2, width: int = 70) -> str:
    """Up to `limit` matched oncology terms with surrounding text, for screening."""
    out = []
    for m in ONC_RE.finditer(text):
        out.append("..." + text[max(0, m.start() - width):m.end() + width].replace("\n", " ") + "...")
        if len(out) == limit:
            break
    return " || ".join(out)


def write_pool(df: pd.DataFrame, out_dir: str) -> None:
    pool = df[df["onc_vignette"] | df["onc_correct"]].copy()
    pool = pool.sort_values("id").reset_index(drop=True)
    pool["context"] = pool["sent1"].map(term_contexts)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cols = ["id", "split", "label", "correct", "stem", "context", "onc_vignette", "onc_correct", "mgmt_stem"]
    pool[cols].to_csv(out / "candidate_pool.csv", index=False)
    with open(out / "screening_view.txt", "w", encoding="utf-8") as f:
        for _, r in pool.iterrows():
            f.write(f"{r['id']} | Q: {r['stem'][:160]} | A: {r['correct'][:80]} | CTX: {r['context'][:330]}\n")
    print(f"\npool written: {len(pool)} items -> {out}")


def main(root: str, pool_out: str | None = None) -> None:
    frames = []
    for split in ("train", "validation", "test"):
        df = pd.read_parquet(Path(root) / f"{split}.parquet")
        df["split"] = split
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    print("columns:", list(df.columns))
    print("rows per split:", df["split"].value_counts().to_dict(), "total", len(df))
    print("sent2 distinct values:", df["sent2"].nunique(), repr(df["sent2"].iloc[0])[:80])

    endings = ["ending0", "ending1", "ending2", "ending3"]
    df["correct"] = df.apply(lambda r: r[endings[int(r["label"])]], axis=1)
    df["stem"] = df["sent1"].map(last_sentence)
    df["onc_vignette"] = df["sent1"].map(patient_oncology)
    df["onc_correct"] = df["correct"].map(lambda x: bool(ONC_RE.search(x)))
    df["onc_any_option"] = df[endings].apply(lambda r: any(ONC_RE.search(x) for x in r), axis=1)
    df["mgmt_stem"] = df["stem"].map(lambda x: bool(MGMT_RE.search(x)))

    rules = {
        "R1 oncology term in vignette": df["onc_vignette"],
        "R2 oncology term in vignette or any option": df["onc_vignette"] | df["onc_any_option"],
        "R3 R1 and management/treatment stem": df["onc_vignette"] & df["mgmt_stem"],
        "R4 oncology term in correct answer": df["onc_correct"],
        "R5 (R1 and management stem) or R4": (df["onc_vignette"] & df["mgmt_stem"]) | df["onc_correct"],
    }
    for name, mask in rules.items():
        sub = df[mask]
        print(f"{name}: {len(sub)}  by split {sub['split'].value_counts().to_dict()}")

    r3 = df[df["onc_vignette"] & df["mgmt_stem"]]
    print("\nlabel balance in R3:", r3["label"].value_counts().sort_index().to_dict())
    print("R3 vignette length (words): median", int(r3["sent1"].str.split().str.len().median()))
    for name in ("R3 R1 and management/treatment stem", "R5 (R1 and management stem) or R4"):
        sub = df[rules[name]]
        print(f"\n20 random examples of {name} (seed 1): matched term | stem | correct answer")
        for _, r in sub.sample(min(20, len(sub)), random_state=1).iterrows():
            term = ONC_RE.search(r["sent1"] + " " + r["correct"])
            print(f"- [{r['id']}] ({term.group(0) if term else '-'}) {r['stem'][:90]} | {r['correct'][:55]}")

    if pool_out:
        write_pool(df, pool_out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
