"""Module 2: read the answer letter from a model output, and combine sampled runs by majority vote.

Rules come from the protocol's `parsing` section (frozen with the protocol):
- Parser: the LAST match of the primary pattern ("Answer: X"); if none, the last match of the fallback
  ("answer is X"); otherwise None (unparseable). Only a capital A-D standing alone counts, so
  "Answer: a combination ..." or "Answer: Doxorubicin" give None.
- Vote: the most frequent letter among the runs; unparseable runs do not vote; a tie goes to the letter
  of the EARLIEST run among the tied letters; if every run is unparseable the vote is None.
- Modal share: votes for the winning letter / ALL runs (unparseable runs count in the denominator).
"""
import re
from collections import Counter
from dataclasses import dataclass


class AnswerParser:
    def __init__(self, parsing_cfg: dict):
        self.primary = re.compile(parsing_cfg["primary_regex"])
        self.fallback = re.compile(parsing_cfg["fallback_regex"])

    def __call__(self, text: str) -> str | None:
        found = self.primary.findall(text) or self.fallback.findall(text)
        return found[-1] if found else None


@dataclass(frozen=True)
class Vote:
    letter: str | None     # majority letter, or None if every run was unparseable
    count: int             # runs that gave the winning letter
    n_runs: int            # all runs, parseable or not
    tie: bool              # True if the top count was shared and the earliest-run rule decided

    @property
    def modal_share(self) -> float:
        return self.count / self.n_runs if self.n_runs else 0.0


def majority_vote(letters: list[str | None]) -> Vote:
    counts = Counter(x for x in letters if x is not None)
    if not counts:
        return Vote(None, 0, len(letters), False)
    top = max(counts.values())
    tied = {letter for letter, c in counts.items() if c == top}
    winner = next(x for x in letters if x in tied)      # earliest run among the tied letters
    return Vote(winner, top, len(letters), len(tied) > 1)


def majority_of_first_k(letters: list[str | None], k: int) -> Vote:
    """Vote over runs 1..k only (for the majority-of-k efficiency curve)."""
    if not 1 <= k <= len(letters):
        raise ValueError(f"k={k} outside 1..{len(letters)}")
    return majority_vote(letters[:k])
