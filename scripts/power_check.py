"""Power check for the paired comparison majority-vote vs greedy (McNemar), Prompt 05.

Minimum detectable accuracy difference (MDE) at 80% power, two-sided alpha 0.05, for several
discordance rates psi = P(the two methods disagree on correctness). Closed-form (Connor 1987),
checked by simulation with an exact McNemar test.
Usage: python -I scripts/power_check.py
"""
import numpy as np
from scipy.stats import binomtest, norm

ALPHA, POWER = 0.05, 0.80
Z_A, Z_B = norm.ppf(1 - ALPHA / 2), norm.ppf(POWER)


def mde(n: int, psi: float) -> float:
    """Smallest delta = p10 - p01 with the target power (Connor 1987), by bisection."""
    lo, hi = 1e-6, psi
    for _ in range(100):
        d = (lo + hi) / 2
        need = (Z_A * np.sqrt(psi) + Z_B * np.sqrt(psi - d * d)) ** 2 / d ** 2
        lo, hi = (d, hi) if need > n else (lo, d)
    return hi


def sim_power(n: int, psi: float, delta: float, reps: int = 4000, seed: int = 1) -> float:
    rng = np.random.default_rng(seed)
    p10, p01 = (psi + delta) / 2, (psi - delta) / 2
    hits = 0
    for _ in range(reps):
        b, c, _ = rng.multinomial(n, [p10, p01, 1 - psi])
        if b + c and binomtest(b, b + c, 0.5).pvalue < ALPHA:
            hits += 1
    return hits / reps


def main() -> None:
    print("n    psi   MDE(points)  simulated power at MDE")
    for n in (684, 450, 300, 120):
        for psi in (0.08, 0.12, 0.16, 0.20):
            d = mde(n, psi)
            print(f"{n:<4} {psi:.2f}  {100 * d:5.1f}        {sim_power(n, psi, d):.2f}")
    for n in (684, 120):
        half = 1.96 * np.sqrt(0.3 * 0.7 / n)
        print(f"95% CI half-width for a proportion of 0.30 with n={n}: +/- {100 * half:.1f} points")


if __name__ == "__main__":
    main()
