"""CPU environment test: runs on the Windows PC and on Kaggle without a GPU.

Checks the libraries import, the protocol config parses, the answer parser behaves as specified, and
the frozen split files match the hashes in the config.
Usage (from repo/): python -I tests/test_cpu_env.py
"""
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    return ok


def main() -> int:
    results = []
    import numpy, pandas, pyarrow, scipy, yaml  # noqa: E401
    print("python", sys.version.split()[0], "| numpy", numpy.__version__, "| pandas", pandas.__version__,
          "| pyarrow", pyarrow.__version__, "| scipy", scipy.__version__, "| pyyaml", yaml.__version__)

    cfg = yaml.safe_load((ROOT / "configs" / "protocol_v1.yaml").read_text(encoding="utf-8"))
    results.append(check("config parses", cfg["model"]["name"] == "Qwen/Qwen2.5-7B-Instruct",
                         f"status {cfg['status']}"))

    primary = re.compile(cfg["parsing"]["primary_regex"])
    fallback = re.compile(cfg["parsing"]["fallback_regex"])

    def parse(text: str):
        m = primary.findall(text) or fallback.findall(text)
        return m[-1] if m else None

    cases = {
        "...so B.\nAnswer: C": "C", "Answer: (D)": "D", "Answer: a combination of drugs": None,
        "The answer is B.": "B", "Answer: Doxorubicin": None, "Answer: B)": "B", "Answer:C": "C",
        "no letter at all": None,
    }
    bad = {t: (parse(t), want) for t, want in cases.items() if parse(t) != want}
    results.append(check("answer parser on 8 cases", not bad, str(bad) if bad else ""))

    split = cfg["data"]["split"]
    for name, key in (("debug_ids.txt", "debug_ids_sha256"), ("test_ids.txt", "test_ids_sha256")):
        ids = (ROOT / "data" / "splits" / name).read_text(encoding="utf-8").strip().splitlines()
        digest = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
        results.append(check(f"{name} matches config hash", digest == split[key], f"{len(ids)} ids"))

    ids_d = set((ROOT / "data/splits/debug_ids.txt").read_text().split())
    ids_t = set((ROOT / "data/splits/test_ids.txt").read_text().split())
    results.append(check("debug and test sets do not overlap", not (ids_d & ids_t)))

    print(f"{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
