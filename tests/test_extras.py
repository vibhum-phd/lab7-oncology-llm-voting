"""CPU tests for the protocol-v2 exploratory additions: new arms, probe module, extras pipeline (mock engine).

Usage (from repo/): python -I -X utf8 tests/test_extras.py
"""
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from board_vote.config import load_config  # noqa: E402
from board_vote.data import Item  # noqa: E402
from board_vote.generate import MockEngine, arm_temperature, runs_per_item, run_arm  # noqa: E402
from board_vote.probe import exact_match, hidden_index, normalise, probe_ids, probe_messages  # noqa: E402

results = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    results.append(bool(ok))


def main() -> int:
    v1, v2 = load_config(), load_config(ROOT / "configs" / "protocol_v2.yaml")
    check("v2 equals v1 except version and amendments",
          all(v1[k] == v2[k] for k in v1 if k != "version") and set(v2) - set(v1) == {"amendments_v2"})
    d = v2["decoding"]
    check("arm settings", [runs_per_item(a, d) for a in ("greedy_rerun_full", "sampled_t03", "sampled_t10", "probe")] == [1, 10, 10, 1]
          and [arm_temperature(a, d) for a in ("sampled", "sampled_t03", "sampled_t10", "greedy")] == [0.7, 0.3, 1.0, 0.0])

    # Probe module.
    item = Item("x1", "Which drug causes hearing loss?", ("Cisplatin", "Bleomycin", "Vincristine", "Doxorubicin"), "A")
    msgs, target = probe_messages(item)
    user = msgs[1]["content"]
    check("hidden option = first non-key option", hidden_index(item) == 1 and target == "Bleomycin")
    check("probe prompt hides the target and marks it", "[MISSING]" in user and "Bleomycin" not in user
          and "option B" in user and "Cisplatin" in user)
    check("normalisation", normalise("B. Bleomycin.\nextra line") == "bleomycin" and normalise("  ") == "")
    check("exact match rules", exact_match("Bleomycin", target) and exact_match("b) bleomycin", target)
          and not exact_match("Bleomycin sulfate", target) and not exact_match("", target))
    ids = [f"t{k:03d}" for k in range(633)]
    p1, p2 = probe_ids(ids, 20261102), probe_ids(ids, 20261102)
    check("100 probe items, fixed by seed, sorted", p1 == p2 and len(p1) == 100 and p1 == sorted(p1))

    # New arms through run_arm with the mock engine.
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        small = ids[:12]
        prompts, seeds = [f"p{i}" for i in small], list(range(12))
        for arm, n in (("greedy_rerun_full", 1), ("sampled_t03", 10), ("sampled_t10", 10), ("probe", 1)):
            run_arm(MockEngine(d), arm, small, prompts, seeds, tmp / f"{arm}.jsonl", d, batch_size=5, max_tokens=64)
            recs = [json.loads(x) for x in (tmp / f"{arm}.jsonl").read_text().splitlines()]
            check(f"{arm}: {n} output(s) per item", len(recs) == 12 * n)

        # Whole extras pipeline on the debug split (mock engine), then the extras evaluation.
        raw = tmp / "raw"
        run = lambda *a: subprocess.run([sys.executable, "-I", "-X", "utf8", *a], cwd=ROOT, capture_output=True, text=True)
        r1 = run("scripts/run_generation.py", "--split", "debug", "--out", str(raw), "--engine", "mock", "--batch-size", "10")
        r2 = run("scripts/run_generation.py", "--split", "debug", "--out", str(raw), "--engine", "mock", "--batch-size", "10",
                 "--config", "configs/protocol_v2.yaml", "--arms", "greedy_rerun_full,sampled_t03,sampled_t10,probe")
        r0 = run("scripts/run_generation.py", "--split", "debug", "--out", str(raw), "--engine", "mock", "--arms", "probe")
        check("v2 arms refused without the v2 config", r0.returncode == 2 and "protocol_v2" in r0.stdout, r0.stdout.strip())
        r3 = run("scripts/run_extras_evaluation.py", "--split", "debug", "--raw", str(raw), "--out", str(tmp / "eval"), "--n-boot", "300")
        ok = r1.returncode == r2.returncode == r3.returncode == 0
        rows = list(csv.DictReader((tmp / "eval" / "results_v2_exploratory.csv").open())) if ok else []
        analyses = {r["analysis"] for r in rows}
        check("extras pipeline runs end to end", ok and {"greedy_self_agreement", "temperature_0.3", "temperature_0.7",
                                                        "temperature_1.0", "contamination_probe"} <= analyses,
              f"{len(rows)} rows" if ok else (r1.stderr + r2.stderr + r3.stderr)[-500:])
        check("exploratory figure 5 written (PNG and PDF)",
              all((tmp / "eval" / "figures" / f"fig5_temperature_exploratory.{x}").exists() for x in ("png", "pdf")))

        # Tables and error patterns run end to end on the same mock outputs.
        ev = str(tmp / "eval")
        r4 = run("scripts/run_evaluation.py", "--split", "debug", "--raw", str(raw), "--out", ev, "--n-boot", "300")
        r5 = run("scripts/make_tables.py", "--results", ev)
        r6 = run("scripts/error_patterns.py", "--split", "debug", "--raw", str(raw), "--results", ev)
        made = [tmp / "eval" / "tables" / "tables.md", tmp / "eval" / "error_patterns.csv", tmp / "eval" / "error_example_ids.csv"]
        check("tables and error patterns run end to end", r4.returncode == r5.returncode == r6.returncode == 0
              and all(p.exists() for p in made), (r4.stderr + r5.stderr + r6.stderr)[-500:])

        # Paper assets (the mock outputs fail the audit's revision check; the report is still written).
        run("scripts/audit_outputs.py", "--split", "debug", "--raw", str(raw), "--out", ev)
        r8 = run("scripts/posthoc_review.py", "--split", "debug", "--raw", str(raw), "--results", ev)
        check("post hoc review analyses run end to end", r8.returncode == 0 and (tmp / "eval" / "posthoc_review.csv").exists(),
              r8.stderr[-500:])
        r7 = run("scripts/make_paper_assets.py", "--results", ev, "--paper", str(tmp / "paper"))
        gen = tmp / "paper" / "generated"
        check("paper assets: number macros and 5 tables generated", r7.returncode == 0 and (gen / "numbers.tex").exists()
              and len(list(gen.glob("tab_*.tex"))) == 5, r7.stderr[-500:])

    print(f"{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
