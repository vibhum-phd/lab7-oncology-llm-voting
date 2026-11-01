"""Generate all arms for one split (modules 1-3 together). GPU run on Kaggle; mock engine for CPU tests.

Usage (from repo/):
  python scripts/run_generation.py --split debug --out results/raw/debug            # real engine (Kaggle)
  python scripts/run_generation.py --split debug --out <tmp> --engine mock          # CPU pipeline test
Arms run in protocol order: greedy, sampled, greedy_rerun (first 50 sorted IDs of the split).
Resumable: rerunning the same command skips items already complete in each arm's file.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from board_vote.config import load_config  # noqa: E402
from board_vote.data import load_medqa, read_ids, select_items  # noqa: E402
from board_vote.generate import MockEngine, VLLMEngine, item_seed, run_arm  # noqa: E402
from board_vote.prompts import build_messages, render_prompt  # noqa: E402

RERUN_N = 50


class StubTokenizer:
    """Chat formatting for the mock engine (the real run uses the model's own tokenizer)."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return text + ("<|im_start|>assistant\n" if add_generation_prompt else "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["debug", "test"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw-dir", default=str(ROOT / "data" / "raw" / "medqa_4opt_hf"))
    ap.add_argument("--engine", choices=["vllm", "mock"], default="vllm")
    ap.add_argument("--arms", default="greedy,sampled,greedy_rerun")
    ap.add_argument("--batch-size", type=int, default=64)
    a = ap.parse_args()

    cfg = load_config()
    if a.split == "test" and a.engine == "vllm" and cfg["status"] != "FROZEN":
        print(f"refusing to run the test split: protocol status is {cfg['status']}, not FROZEN")
        return 2
    ids = sorted(read_ids(ROOT / "data" / "splits" / f"{a.split}_ids.txt"))
    items = select_items(load_medqa(a.raw_dir), ids)
    seeds = [item_seed(cfg["model"]["engine_args"]["seed"], k) for k in range(len(ids))]
    if a.engine == "vllm":
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg["model"]["name"], revision=cfg["model"]["revision"])
        engine = VLLMEngine(cfg["model"], cfg["decoding"])
        import vllm
        meta = {"engine": f"vllm=={vllm.__version__}", "model_revision": cfg["model"]["revision"]}
    else:
        tok, engine, meta = StubTokenizer(), MockEngine(cfg["decoding"]), {"engine": "mock"}
    prompts = [render_prompt(build_messages(it, cfg["prompt"]), tok) for it in items]

    out = Path(a.out)
    summaries, t0 = [], time.time()
    for arm in a.arms.split(","):
        sel = slice(0, RERUN_N) if arm == "greedy_rerun" else slice(None)
        s = run_arm(engine, arm, ids[sel], prompts[sel], seeds[sel], out / f"{arm}.jsonl", cfg["decoding"],
                    batch_size=a.batch_size, meta=meta)
        print(json.dumps(s))
        summaries.append(s)
    summary = {"split": a.split, "engine": meta["engine"], "protocol_status": cfg["status"],
               "n_items": len(ids), "arms": summaries, "total_seconds": round(time.time() - t0, 1)}
    (out / "run_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
