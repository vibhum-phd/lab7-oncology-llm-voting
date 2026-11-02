"""CPU test: the released results can be recomputed exactly from the released parsed answers (no GPU, no raw outputs,
no MedQA text). Uses the committed results/item_level.csv and results/results.csv.

Usage (from repo/): python -I -X utf8 tests/test_reproduce.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
results = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    results.append(bool(ok))


def main() -> int:
    r = subprocess.run([sys.executable, "-I", "-X", "utf8", "scripts/recompute_from_item_level.py"], cwd=ROOT,
                       capture_output=True, text=True)
    last = (r.stdout.strip().splitlines() or [""])[0]
    check("results.csv recomputed exactly from item_level.csv", r.returncode == 0 and "0 differences" in last,
          last or r.stderr[-300:])
    print(f"{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
