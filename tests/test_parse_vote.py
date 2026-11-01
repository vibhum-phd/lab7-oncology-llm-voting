"""CPU tests for module 2 (answer parsing and majority vote).

Part A: rule cases written from the protocol. Part B (optional): re-parse the 300 real debug-set
outputs from Kaggle run 01 v2 and check our parser agrees with the letters parsed on Kaggle.
Usage (from repo/): python -I -X utf8 tests/test_parse_vote.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from board_vote.config import load_config  # noqa: E402
from board_vote.parse import AnswerParser, majority_of_first_k, majority_vote  # noqa: E402

DEBUG_RUN = ROOT / "results" / "env" / "kaggle_01_v2" / "debug_generations.jsonl"
results = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    results.append(bool(ok))


def main() -> int:
    parse = AnswerParser(load_config()["parsing"])

    # A1. Parser rules.
    cases = {
        "Step 1 ... so B looks right.\nAnswer: C": "C",            # final line wins over earlier mentions
        "Answer: A\nWait, re-checking.\nAnswer: D": "D",          # last "Answer:" wins
        "Answer: (B)": "B", "Answer: B)": "B", "Answer:C": "C", "answer: D": "D",
        "**Answer: C**": "C",                                       # bold markers around the whole line
        "The answer is B.": "B",                                    # fallback pattern
        "Answer: a combination of chemotherapy": None,             # lowercase word, not an option
        "Answer: Doxorubicin": None,                                # word starting with D
        "Answer: X": None,                                          # the instruction's placeholder
        "Answer: **C**": "C",                                       # bold letter (gap closed before freezing)
        "**Final Answer**: D": "D",                                 # real debug output, run 02
        "Answer: **Doxorubicin**": None,                            # bold word, still not an option
        "I think the best option is C.": None,                      # no answer line at all
        "": None,
    }
    wrong = {t: (parse(t), w) for t, w in cases.items() if parse(t) != w}
    check(f"parser on {len(cases)} rule cases", not wrong, str(wrong) if wrong else "")

    # A2. Vote rules.
    v = majority_vote(["A", "B", "A", None, "C", "A", "B", "B", "A", "D"])
    check("clear majority", (v.letter, v.count, v.tie) == ("A", 4, False), f"{v}")
    check("modal share counts unparseable runs in the denominator", v.modal_share == 0.4)
    v = majority_vote(["B", "A", "A", "B"])
    check("tie -> earliest run among tied letters", (v.letter, v.tie) == ("B", True), f"{v}")
    v = majority_vote([None, "C", "D", "D", "C"])
    check("tie with unparseable first run", v.letter == "C", f"{v}")
    v = majority_vote([None] * 10)
    check("all unparseable -> None", (v.letter, v.count, v.modal_share) == (None, 0, 0.0))
    v = majority_vote(["D"] * 10)
    check("unanimous", (v.letter, v.modal_share, v.tie) == ("D", 1.0, False))
    runs = ["A", "B", "B", "A", "A"]
    check("majority of first k", [majority_of_first_k(runs, k).letter for k in (1, 3, 5)] == ["A", "B", "A"])
    try:
        majority_of_first_k(runs, 6)
        check("k larger than runs refused", False)
    except ValueError:
        check("k larger than runs refused", True)

    # B. Real outputs from the Kaggle debug run (file is local only; skipped on a fresh clone).
    if DEBUG_RUN.exists():
        rows = [json.loads(line) for line in DEBUG_RUN.open(encoding="utf-8")]
        disagree = [r["id"] for r in rows if parse(r["text"]) != r["parsed"]]
        check("re-parse of 300 real debug outputs matches Kaggle", len(rows) == 300 and not disagree,
              f"{len(rows)} outputs, {len(disagree)} disagreements")
        sampled = {}
        for r in rows:
            if r["arm"] == "sampled":
                sampled.setdefault(r["id"], [None] * 10)[r["run"]] = parse(r["text"])
        votes = {i: majority_vote(x) for i, x in sampled.items()}
        ties = sum(v.tie for v in votes.values())
        shares = sorted(round(v.modal_share, 1) for v in votes.values())
        check("votes computed for all 25 debug items", len(votes) == 25,
              f"ties broken by the earliest-run rule: {ties}; modal shares: {shares}")
    else:
        print("[SKIP] real debug outputs not present (local-only file)")

    print(f"{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
