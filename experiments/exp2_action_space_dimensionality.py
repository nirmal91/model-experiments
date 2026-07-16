"""Experiment 2 — Why manipulation (6-DoF) is drastically harder than driving (~4-DoF).

Claim under test: the action space for manipulation is ~6-dimensional vs ~4
for self-driving, and "complexity scales drastically" with dimension — one of
the reasons end-to-end learning that works for driving doesn't yet transfer
to manipulation.

Two measurements:
1. Coverage: number of resolution-r cells needed to tile a d-dim action
   space (r^d) — a proxy for how much data an end-to-end learner needs to
   have "seen something nearby".
2. Exploration: empirical samples until a uniform random policy first lands
   within tolerance eps of a target configuration, measured in d = 2..7.
   P(hit) = eps^d per draw, so expected samples grow exponentially in d.
"""

import numpy as np
import matplotlib.pyplot as plt

import plot_style
from plot_style import BLUE, GREEN, INK_2

rng = np.random.default_rng(11)

EPS = 0.25          # tolerance per dimension (unit action cube)
DIMS = range(2, 8)
TRIALS = 400
MAX_DRAWS = 2_000_000


def samples_to_first_hit(d: int) -> float:
    """Median random draws until all |x_i - target_i| < EPS/2."""
    target = np.full(d, 0.5)
    counts = []
    for _ in range(TRIALS):
        n, batch = 0, 4096
        while n < MAX_DRAWS:
            pts = rng.random((batch, d))
            hits = np.all(np.abs(pts - target) < EPS / 2, axis=1)
            idx = np.argmax(hits)
            if hits[idx]:
                counts.append(n + idx + 1)
                break
            n += batch
        else:
            counts.append(MAX_DRAWS)
    return float(np.median(counts))


def main():
    plot_style.apply()
    dims = np.array(list(DIMS))
    measured = np.array([samples_to_first_hit(d) for d in dims])
    expected = 1.0 / (EPS ** dims)  # E[draws] for a geometric with p = eps^d

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dims, expected, color=INK_2, linestyle=":", linewidth=1.2)
    ax.annotate("theory 1/ε^d", (dims[-2], expected[-2] * 1.6),
                color=INK_2, fontsize=8)
    ax.plot(dims, measured, color=BLUE, marker="o", markersize=6)
    ax.annotate("measured (median of 400 runs)", (2.6, 20),
                color=BLUE, fontsize=9)
    ax.set_yscale("log")
    ax.axvline(4, color=GREEN, linewidth=1)
    ax.annotate("driving ≈ 4-DoF", (4.07, 9000), color=GREEN, fontsize=9)
    ax.axvline(6, color=GREEN, linewidth=1)
    ax.annotate("manipulation ≈ 6-DoF", (6.07, 9000), color=GREEN, fontsize=9)
    ax.set_xlabel("Action-space dimension (d)")
    ax.set_ylabel("Random samples to first success (log scale)")
    ax.set_title("Exploration cost grows exponentially with action dimension",
                 loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig("results/exp2_action_space_dimensionality.png")

    print(f"{'d':>3} {'expected 1/eps^d':>18} {'measured median':>16}")
    for d, e, m in zip(dims, expected, measured):
        print(f"{d:>3} {e:>18,.0f} {m:>16,.0f}")
    r = expected[dims == 6][0] / expected[dims == 4][0]
    print(f"\n4-DoF -> 6-DoF at eps={EPS}: {r:,.0f}x more exploration per hit.")
    print(f"Grid coverage at 10 bins/dim: 10^4 = 10,000 vs 10^6 = 1,000,000 cells (100x).")


if __name__ == "__main__":
    main()
