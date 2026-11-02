"""Exploratory contamination probe (protocol v2): can the model reproduce a hidden answer option word for word?

For each probe item, one wrong option (the first non-key option in A-D order) is replaced by [MISSING],
and the model is asked to write that option's exact text. Scoring is exact match after normalisation.
A high match rate suggests the item text was memorised; a low rate does not prove it was not.
"""
import random
import re
import string

from .data import LETTERS, Item

N_ITEMS = 100
SYSTEM = "You are completing a multiple-choice exam question that has one option missing."
INSTRUCTION = ("One option above is shown as [MISSING]. Write only the exact original text of option {letter}, "
               "with no explanation.")


def probe_ids(sorted_test_ids: list[str], seed: int, n: int = N_ITEMS) -> list[str]:
    ids = list(sorted_test_ids)
    random.Random(seed).shuffle(ids)
    return sorted(ids[:n])


def hidden_index(item: Item) -> int:
    return next(k for k, letter in enumerate(LETTERS) if letter != item.key)


def probe_messages(item: Item) -> tuple[list[dict], str]:
    """Messages for the probe and the hidden option's text (the target)."""
    h = hidden_index(item)
    lines = [f"{L}. {'[MISSING]' if k == h else text}" for k, (L, text) in enumerate(zip(LETTERS, item.options))]
    user = f"{item.question}\n\nOptions:\n" + "\n".join(lines) + "\n\n" + INSTRUCTION.format(letter=LETTERS[h])
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], item.options[h]


def normalise(text: str) -> str:
    text = text.strip().splitlines()[0] if text.strip() else ""
    text = re.sub(r"^(option\s+)?[ABCD][.):]\s*", "", text, flags=re.IGNORECASE)
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def exact_match(output: str, target: str) -> bool:
    return normalise(output) == normalise(target) and normalise(target) != ""
