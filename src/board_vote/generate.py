"""Module 3: generate model outputs for one arm, saving as it goes so an interrupted run can resume.

Arms (protocol `decoding` section):
- "greedy": temperature 0, 1 output per item.
- "sampled": temperature 0.7, top-p 0.8, top-k 20, 10 outputs per item, seed = base + item index.
- "greedy_rerun": greedy again on a fixed subset, to check determinism.

Output: one JSON line per output in results/raw/{arm}.jsonl (git-ignored: the text can quote items).
Resuming: an item counts as done only when all its outputs are in the file. A half-written last line or
an item with too few outputs (session killed mid-write) is removed before continuing.
The engine is either VLLMEngine (Kaggle GPU) or MockEngine (CPU tests); both return the same shape.
"""
import json
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ARM_RUNS = {"greedy": 1, "greedy_rerun": 1}   # "sampled" uses decoding.sampled.n


@dataclass(frozen=True)
class Completion:
    text: str
    new_tokens: int
    finish: str            # "stop" (ended normally) or "length" (hit max_tokens)


def runs_per_item(arm: str, decoding_cfg: dict) -> int:
    return decoding_cfg["sampled"]["n"] if arm == "sampled" else ARM_RUNS[arm]


def item_seed(base: int, index: int) -> int:
    """Protocol seed rule: base seed + the item's index in the sorted ID list."""
    return base + index


class VLLMEngine:
    """Real engine (GPU, Linux only). Settings come from the protocol config, never from code."""

    def __init__(self, model_cfg: dict, decoding_cfg: dict):
        from vllm import LLM, SamplingParams        # imported here so CPU tests never need vLLM
        self._params = SamplingParams
        self.decoding = decoding_cfg
        self.llm = LLM(model=model_cfg["name"], revision=model_cfg["revision"], dtype=model_cfg["dtype"],
                       **model_cfg["engine_args"])

    def generate(self, prompts: list[str], arm: str, seeds: list[int]) -> list[list[Completion]]:
        d = self.decoding
        if arm == "sampled":
            s = d["sampled"]
            params = [self._params(n=s["n"], temperature=s["temperature"], top_p=s["top_p"], top_k=s["top_k"],
                                   max_tokens=d["max_tokens"], repetition_penalty=d["repetition_penalty"],
                                   seed=seed) for seed in seeds]
        else:
            params = [self._params(temperature=d["greedy"]["temperature"], max_tokens=d["max_tokens"],
                                   repetition_penalty=d["repetition_penalty"])] * len(prompts)
        outs = self.llm.generate(prompts, params)
        return [[Completion(c.text, len(c.token_ids), c.finish_reason) for c in o.outputs] for o in outs]


class MockEngine:
    """Fake model for CPU tests: deterministic given the seed, same output shape as VLLMEngine."""

    def __init__(self, decoding_cfg: dict, fail_after_batches: int | None = None):
        self.decoding = decoding_cfg
        self.fail_after = fail_after_batches
        self.batches = 0

    def generate(self, prompts, arm, seeds):
        if self.fail_after is not None and self.batches >= self.fail_after:
            raise RuntimeError("simulated session stop")
        self.batches += 1
        n = runs_per_item(arm, self.decoding)
        out = []
        for prompt, seed in zip(prompts, seeds):
            rng = random.Random(f"{arm}-{seed}-{len(prompt)}")
            out.append([Completion(f"Reasoning step.\nAnswer: {rng.choice('ABCD')}", rng.randint(50, 500), "stop")
                        for _ in range(n)])
        return out


def load_done(path: Path, n_expected: int) -> set[str]:
    """IDs whose outputs are complete. Rewrites the file without broken lines or incomplete items."""
    if not path.exists():
        return set()
    groups, broken = defaultdict(list), 0
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
            groups[rec["id"]].append(rec)
        except (json.JSONDecodeError, KeyError):
            broken += 1
    complete = {i for i, recs in groups.items() if sorted(r["run"] for r in recs) == list(range(n_expected))}
    if broken or len(complete) < len(groups):
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for i in groups:
                if i in complete:
                    for rec in sorted(groups[i], key=lambda r: r["run"]):
                        f.write(json.dumps(rec) + "\n")
        os.replace(tmp, path)
    return complete


def run_arm(engine, arm: str, ids: list[str], prompts: list[str], seeds: list[int], out_path: Path,
            decoding_cfg: dict, batch_size: int = 64, meta: dict | None = None) -> dict:
    """Generate `arm` for every id not yet complete in out_path; append after each batch."""
    n = runs_per_item(arm, decoding_cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path, n)
    todo = [k for k, i in enumerate(ids) if i not in done]
    written, t_start = 0, time.time()
    for start in range(0, len(todo), batch_size):
        batch = todo[start:start + batch_size]
        t = time.time()
        comps = engine.generate([prompts[k] for k in batch], arm, [seeds[k] for k in batch])
        seconds = round(time.time() - t, 2)
        if len(comps) != len(batch) or any(len(c) != n for c in comps):
            raise RuntimeError(f"engine returned the wrong number of outputs for arm {arm}")
        with out_path.open("a", encoding="utf-8") as f:
            for k, item_comps in zip(batch, comps):
                for run, c in enumerate(item_comps):
                    f.write(json.dumps({"id": ids[k], "arm": arm, "run": run, "seed": seeds[k], "text": c.text,
                                        "new_tokens": c.new_tokens, "finish": c.finish,
                                        "batch_seconds": seconds, "batch_items": len(batch), **(meta or {})}) + "\n")
            f.flush()
            os.fsync(f.fileno())
        written += len(batch)
    return {"arm": arm, "items_total": len(ids), "already_done": len(done), "generated_now": written,
            "seconds": round(time.time() - t_start, 1)}
