"""CPU tests for module 1 (data loading, validation, prompt building). No GPU, no model download.

Needs the MedQA parquet files in data/raw/medqa_4opt_hf/ (see data/medqa_source.md).
Usage (from repo/): python -I -X utf8 tests/test_data_prompts.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from board_vote.config import load_config  # noqa: E402
from board_vote.data import load_medqa, read_ids, select_items  # noqa: E402
from board_vote.prompts import build_messages, render_prompt  # noqa: E402


class StubTokenizer:
    """Stands in for the Hugging Face tokenizer: a visible, Qwen-like chat format."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return text + ("<|im_start|>assistant\n" if add_generation_prompt else "")


results = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    results.append(bool(ok))


def raises(fn, *args):
    try:
        fn(*args)
    except ValueError as e:
        return str(e)
    return None


def main() -> int:
    cfg = load_config()
    df = load_medqa(ROOT / "data" / "raw" / "medqa_4opt_hf")
    check("MedQA loads", len(df) == 12723, f"{len(df)} items")

    debug_ids = read_ids(ROOT / "data" / "splits" / "debug_ids.txt")
    test_ids = read_ids(ROOT / "data" / "splits" / "test_ids.txt")
    debug = select_items(df, debug_ids)
    test = select_items(df, test_ids)
    check("all 25 debug items valid", len(debug) == 25)
    check("all 633 test items valid", len(test) == 633)
    check("order kept", [i.id for i in test] == test_ids)
    keys = pd.Series([i.key for i in test]).value_counts().sort_index().to_dict()
    check("keys are A-D", set(keys) <= set("ABCD"), f"test key counts {keys}")

    # Negative cases: the validator must refuse bad input.
    check("unknown id refused", raises(select_items, df, ["no-such-id"]) is not None)
    bad = df.loc[[debug_ids[0]]].copy()
    bad.loc[debug_ids[0], "label"] = 7
    check("label outside 0-3 refused", raises(select_items, bad, [debug_ids[0]]) is not None)
    bad = df.loc[[debug_ids[0]]].copy()
    bad.loc[debug_ids[0], "ending2"] = "  "
    check("empty option refused", raises(select_items, bad, [debug_ids[0]]) is not None)

    # Prompt building.
    item = debug[0]
    msgs = build_messages(item, cfg["prompt"])
    user = msgs[1]["content"]
    check("two messages: system, user", [m["role"] for m in msgs] == ["system", "user"])
    check("question text included verbatim", item.question in user)
    check("options labelled A-D in dataset order",
          all(f"{L}. {o}\n" in user for L, o in zip("ABCD", item.options)))
    check("ends with the answer-format instruction", user.rstrip().endswith("one of A, B, C or D."))
    check("no unfilled {placeholders}", "{" not in user.replace(item.question, ""))
    prompt = render_prompt(msgs, StubTokenizer())
    check("prompt ends with assistant turn opener", prompt.endswith("<|im_start|>assistant\n"))
    lengths = [len(build_messages(i, cfg["prompt"])[1]["content"]) for i in test]
    check("every test item builds a prompt", len(lengths) == 633,
          f"user message chars: min {min(lengths)}, median {sorted(lengths)[316]}, max {max(lengths)}")

    print(f"{sum(results)}/{len(results)} checks passed")
    if "--show" in sys.argv:
        print("\n===== exact prompt for debug item", item.id, "(stub chat format) =====\n" + prompt)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
