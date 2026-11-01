"""Module 5b: the paper's figures, drawn only from the metric and item tables (never from typed numbers).

Fig 2: majority-of-k accuracy vs k (left) and output tokens per item vs k (right), greedy as a reference.
Fig 3: accuracy vs coverage when answering only items with >= x of 10 runs agreeing.
Fig 4: distribution of modal share, management (M) vs other oncology (O) items.
Static PNG + PDF for print. Colours: validated reference palette, light mode (slot 1 blue, slot 2 orange).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
K_CURVE = [1, 3, 5, 7, 9, 10]


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def _metric(res: pd.DataFrame, name: str, subset: str = "all") -> float:
    return float(res.loc[(res.metric == name) & (res.subset == subset), "value"].iloc[0])


def _save(fig, out: Path, name: str) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{name}.png", out / f"{name}.pdf"]
    fig.savefig(paths[0], dpi=200, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def fig_majority_of_k(res: pd.DataFrame, items: pd.DataFrame, n_runs: int, out: Path) -> list[Path]:
    acc = [_metric(res, f"accuracy_majority_of_first_{k}") for k in K_CURVE]
    greedy = _metric(res, "accuracy_greedy")
    tok_per_run = items["sampled_tokens"].mean() / n_runs
    tokens = [k * tok_per_run for k in K_CURVE]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.8))
    a1.plot(K_CURVE, acc, color=BLUE, linewidth=1.5, marker="o", markersize=5, label="majority of first k")
    a1.axhline(greedy, color=INK2, linewidth=1.2, linestyle="--")
    a1.annotate("greedy (1 run, T = 0)", (K_CURVE[-1], greedy), textcoords="offset points", xytext=(0, 4),
                ha="right", fontsize=7, color=INK2)
    a1.annotate(f"{acc[-1]:.3f}", (K_CURVE[-1], acc[-1]), textcoords="offset points", xytext=(4, -3), fontsize=7, color=INK)
    a1.set_xlabel("sampled runs voted on (k)", fontsize=8)
    a1.set_ylabel("accuracy", fontsize=8)
    a1.set_xticks(K_CURVE)
    a2.plot(K_CURVE, tokens, color=BLUE, linewidth=1.5, marker="o", markersize=5)
    a2.axhline(items["greedy_tokens"].mean(), color=INK2, linewidth=1.2, linestyle="--")
    a2.annotate("greedy", (K_CURVE[-1], items["greedy_tokens"].mean()), textcoords="offset points",
                xytext=(0, 4), ha="right", fontsize=7, color=INK2)
    a2.set_xlabel("sampled runs voted on (k)", fontsize=8)
    a2.set_ylabel("output tokens per item", fontsize=8)
    a2.set_xticks(K_CURVE)
    for ax in (a1, a2):
        _style(ax)
    fig.tight_layout()
    return _save(fig, out, "fig2_majority_of_k")


def fig_accuracy_coverage(res: pd.DataFrame, items: pd.DataFrame, n_runs: int, out: Path) -> list[Path]:
    cov, acc = [], []
    for x in range(1, n_runs + 1):
        sel = items["vote_count"] >= x
        if sel.any():
            cov.append(sel.mean())
            acc.append(items.loc[sel, "vote_correct"].mean())
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    ax.plot(cov, acc, color=BLUE, linewidth=1.5, marker="o", markersize=5)
    for x, (c, a) in enumerate(zip(cov, acc), start=1):
        if x in (1, 5, 9, 10) and x <= len(cov):
            ax.annotate(f"≥{x}/10", (c, a), textcoords="offset points", xytext=(4, 4), fontsize=7, color=INK2)
    ax.set_xlabel("coverage (share of items answered)", fontsize=8)
    ax.set_ylabel("accuracy of answered items", fontsize=8)
    ax.set_xlim(0, 1.02)
    _style(ax)
    fig.tight_layout()
    return _save(fig, out, "fig3_accuracy_coverage")


def fig_modal_share(items: pd.DataFrame, n_runs: int, out: Path) -> list[Path]:
    levels = np.arange(1, n_runs + 1) / n_runs
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    width = 0.038
    for offset, (code, color, label) in zip((-width / 2 - 0.002, width / 2 + 0.002),
                                            (("M", BLUE, "management (M)"), ("O", ORANGE, "other oncology (O)"))):
        sub = items[items["code"] == code]
        if sub.empty:
            continue
        share = [(np.isclose(sub["modal_share"], lv)).mean() for lv in levels]
        ax.bar(levels + offset, share, width=width, color=color, label=f"{label}, n = {len(sub)}")
    ax.set_xlabel("modal share (runs agreeing with the vote, of 10)", fontsize=8)
    ax.set_ylabel("share of items", fontsize=8)
    ax.set_xticks(levels, [f"{int(round(lv * n_runs))}" for lv in levels])
    ax.legend(frameon=False, fontsize=7, labelcolor=INK)
    _style(ax)
    fig.tight_layout()
    return _save(fig, out, "fig4_modal_share")


def make_all(res: pd.DataFrame, items: pd.DataFrame, n_runs: int, out: Path) -> list[Path]:
    return (fig_majority_of_k(res, items, n_runs, out) + fig_accuracy_coverage(res, items, n_runs, out)
            + fig_modal_share(items, n_runs, out))
