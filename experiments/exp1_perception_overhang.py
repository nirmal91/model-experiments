"""Experiment 1 — Why 85% -> 99% perception is an *overhang*, not a +14% bump.

Claim under test (from the conversation): pose-estimation models recently
jumped from ~85% to ~99% per-call reliability, and that jump is worth far
more than it sounds because real lab protocols are LONG SEQUENCES of
perception-dependent steps. In a modular pipeline, per-step reliability
compounds multiplicatively, so protocol-level success is p^N.

We Monte-Carlo a pick-and-place protocol of N perception-gated steps and
compare against the closed form, then compute the longest protocol each
reliability level can run at >=50% and >=90% end-to-end success.
"""

import numpy as np
import matplotlib.pyplot as plt

import plot_style
from plot_style import SERIES, INK_2

rng = np.random.default_rng(7)

LEVELS = [0.85, 0.95, 0.99, 0.999]
MAX_STEPS = 60
TRIALS = 20_000


def simulate(p: float, n_steps: int) -> float:
    """Fraction of protocol runs where every perception-gated step succeeds."""
    steps_ok = rng.random((TRIALS, n_steps)) < p
    return steps_ok.all(axis=1).mean()


def max_protocol_len(p: float, floor: float) -> int:
    """Longest N with p^N >= floor."""
    return int(np.floor(np.log(floor) / np.log(p)))


def main():
    plot_style.apply()
    steps = np.arange(1, MAX_STEPS + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    for color, p in zip(SERIES, LEVELS):
        analytic = p ** steps
        # spot-check the closed form with simulation at a few lengths
        for n in (5, 20, 50):
            sim = simulate(p, n)
            assert abs(sim - p**n) < 0.02, (p, n, sim)
        ax.plot(steps, analytic, color=color)
        ax.annotate(f"{p:.1%} per-step", (steps[-1], analytic[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    color=color, fontsize=9, va="center")

    ax.axhline(0.5, color=INK_2, linewidth=0.8, linestyle=":")
    ax.annotate("50% usable threshold", (2, 0.515), color=INK_2, fontsize=8)
    ax.set_xlim(1, MAX_STEPS + 14)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Perception-gated steps in protocol (N)")
    ax.set_ylabel("End-to-end protocol success  (p^N)")
    ax.set_title("Per-step perception reliability compounds across a protocol",
                 loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig("results/exp1_perception_overhang.png")

    print("Longest protocol each perception level supports:")
    print(f"{'per-step p':>12} {'N @ 50% success':>16} {'N @ 90% success':>16}")
    for p in LEVELS:
        print(f"{p:>12.1%} {max_protocol_len(p, 0.5):>16d} {max_protocol_len(p, 0.9):>16d}")

    n50_85, n50_99 = max_protocol_len(0.85, 0.5), max_protocol_len(0.99, 0.5)
    print(f"\n85% -> 99% multiplies feasible protocol length by "
          f"{n50_99 / n50_85:.0f}x ({n50_85} -> {n50_99} steps at 50% success).")


if __name__ == "__main__":
    main()
