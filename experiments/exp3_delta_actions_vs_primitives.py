"""Experiment 3 — Delta-action control vs parameterized primitives ("function calls").

Claims under test:
1. Framing actions as parameterized function calls (grab(pose), move(pose))
   instead of predicting incremental delta actions lets the stack *leverage
   perception models directly* — so when pose estimation jumps 85% -> 99%,
   task success rides that jump. An end-to-end delta policy carries its own
   implicit perception, so external perception gains don't transfer without
   retraining.
2. Delta-action policies are *shaky*: each step is a fresh noisy inference,
   so the trajectory jitters. A primitive executes one planned path.

Setup: planar reach-and-grasp in a 400mm workspace, 2mm grasp tolerance.
A pose model of accuracy `a` returns the object pose with isotropic Gaussian
error sized so that P(error < tolerance) = a for a single call.

- MODULAR: one pose-model call, plan a straight path to the estimate, execute
  with small actuation noise, close gripper. Success = true error < tolerance.
- END-TO-END (delta): 10Hz loop; each tick the policy infers the target with
  its own fixed implicit perception (~88% single-call equivalent, unchanged
  by external model progress) and steps toward it with actuation noise;
  grasps when it believes it has converged.
"""

import numpy as np
import matplotlib.pyplot as plt

import plot_style
from plot_style import BLUE, GREEN, INK_2

rng = np.random.default_rng(3)

TOL = 2.0            # grasp tolerance, mm
WORKSPACE = 400.0    # mm
EPISODES = 4000
STEP = 25.0          # max delta-action step, mm/tick
ACT_NOISE = 0.4      # actuation noise per executed motion, mm
INFER_NOISE = 0.20   # delta policy's action-head noise, fraction of the step
IMPLICIT_ACC = 0.88  # end-to-end policy's built-in perception quality
MAX_TICKS = 60


def sigma_for_accuracy(a: float) -> float:
    """2D Gaussian error scale with P(||err|| < TOL) = a."""
    return TOL / np.sqrt(-2.0 * np.log(1.0 - a))


def run_modular(acc: float) -> tuple[float, float]:
    """Returns (success rate, mean path-length ratio)."""
    sigma = sigma_for_accuracy(acc)
    target = rng.uniform(0.3 * WORKSPACE, 0.7 * WORKSPACE, (EPISODES, 2))
    est = target + rng.normal(0, sigma, (EPISODES, 2))
    landed = est + rng.normal(0, ACT_NOISE, (EPISODES, 2))
    err = np.linalg.norm(landed - target, axis=1)
    return float((err < TOL).mean()), 1.0  # planned straight path


def run_delta(acc_external_unused: float) -> tuple[float, float, float]:
    """Returns (success rate, mean path-length ratio, mean heading change in
    degrees per tick). Ignores the external pose model: the policy's implicit
    perception is fixed at IMPLICIT_ACC, and its action head adds inference
    noise to every emitted delta."""
    sigma = sigma_for_accuracy(IMPLICIT_ACC)
    succ, ratios, jitters = 0, [], []
    for _ in range(EPISODES // 10):  # slower loop, fewer episodes suffice
        target = rng.uniform(0.3 * WORKSPACE, 0.7 * WORKSPACE, 2)
        pos = np.zeros(2)
        path, start = 0.0, pos.copy()
        headings = []
        for _ in range(MAX_TICKS):
            obs = target + rng.normal(0, sigma, 2)
            vec = obs - pos
            dist = np.linalg.norm(vec)
            if dist < TOL:      # policy believes it converged -> grasp
                break
            move = vec / dist * min(STEP, dist)
            move += rng.normal(0, INFER_NOISE * np.linalg.norm(move), 2)
            move += rng.normal(0, ACT_NOISE, 2)
            headings.append(np.arctan2(move[1], move[0]))
            pos = pos + move
            path += np.linalg.norm(move)
        succ += np.linalg.norm(pos - target) < TOL
        ratios.append(path / np.linalg.norm(target - start))
        if len(headings) > 1:
            dh = np.diff(np.unwrap(headings))
            jitters.append(np.degrees(np.abs(dh)).mean())
    return (succ / (EPISODES // 10), float(np.mean(ratios)),
            float(np.mean(jitters)))


def main():
    plot_style.apply()
    accs = np.array([0.85, 0.90, 0.95, 0.99, 0.995, 0.999])

    modular = [run_modular(a) for a in accs]
    delta = [run_delta(a) for a in accs]
    mod_s = np.array([m[0] for m in modular])
    del_s = np.array([d[0] for d in delta])
    del_ratio = np.mean([d[1] for d in delta])
    del_jitter = np.mean([d[2] for d in delta])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(accs * 100, mod_s, color=BLUE, marker="o", markersize=6)
    ax.annotate("modular: primitives on top of the pose model",
                (85.0, 0.96), color=BLUE, fontsize=9)
    ax.plot(accs * 100, del_s, color=GREEN, marker="o", markersize=6)
    ax.annotate("end-to-end delta policy (implicit perception, fixed)",
                (89.5, 0.74), color=GREEN, fontsize=9)
    ax.set_xlabel("External pose-model accuracy (%)")
    ax.set_ylabel("Grasp success rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Only the modular stack converts perception-model gains into task success",
                 loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig("results/exp3_delta_vs_primitives.png")

    print(f"{'pose acc':>9} {'modular':>9} {'delta':>7}")
    for a, m, d in zip(accs, mod_s, del_s):
        print(f"{a:>9.1%} {m:>9.1%} {d:>7.1%}")
    print(f"\nShakiness — path-length ratio: modular 1.00 (planned straight) "
          f"vs delta {del_ratio:.2f}; heading jitter: modular ~0 deg/tick vs "
          f"delta {del_jitter:.1f} deg/tick.")
    print("Note: at ~85-90% pose accuracy the two are comparable — the closed "
          "delta loop re-observes every tick, which averages out weak "
          "perception. The crossover is the point: once perception is "
          "near-perfect, one planned primitive call beats servoing, and only "
          "the modular stack rides the external model's 85%->99.9% jump.")


if __name__ == "__main__":
    main()
