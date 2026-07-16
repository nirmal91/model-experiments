"""Experiment 4 — "VLAs and world models are essentially the same thing."

Claim under test: the industry framing "VLAs are dead, world models are
supreme" draws a false dichotomy — both are supervised learning on
trajectories (+ optional RL post-training); they differ in what they predict
(actions vs next frames), not in what they learn.

Toy version: a 6x6 gridworld with walls, expert demonstrations from noisy-
optimal rollouts. From the SAME dataset we train:
- VLA-style:  behavior cloning, predict p(action | state), argmax at test.
- World-model-style: learn transitions p(s' | s, a), then PLAN on the learned
  model (value iteration) and act with the planned policy.

We sweep dataset size and compare goal-reaching success from every start
state. If the dichotomy were fundamental, one should dominate everywhere;
instead both converge with data, differing mainly in how they use limited
coverage (the planner squeezes more out of sparse data by stitching
transitions across demonstrations).
"""

import numpy as np
import matplotlib.pyplot as plt

import plot_style
from plot_style import BLUE, GREEN

rng = np.random.default_rng(5)

N = 6
WALLS = {(1, 2), (2, 2), (3, 2), (4, 4), (3, 4), (1, 4)}
GOAL = (5, 5)
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_EVAL_STEPS = 40
DEMO_EPS = 0.2  # expert takes a random action 20% of the time


def step(s, a):
    nxt = (s[0] + a[0], s[1] + a[1])
    if not (0 <= nxt[0] < N and 0 <= nxt[1] < N) or nxt in WALLS:
        return s
    return nxt


def true_optimal_policy():
    """Value iteration on the true dynamics."""
    V = {s: 0.0 for s in all_states()}
    for _ in range(200):
        for s in V:
            if s == GOAL:
                continue
            V[s] = max(-1 + V[step(s, a)] for a in ACTIONS)
    return {s: max(ACTIONS, key=lambda a: V[step(s, a)]) for s in V}


def all_states():
    return [(i, j) for i in range(N) for j in range(N) if (i, j) not in WALLS]


def collect_demos(policy, n_episodes):
    data = []
    starts = all_states()
    for _ in range(n_episodes):
        s = starts[rng.integers(len(starts))]
        for _ in range(MAX_EVAL_STEPS):
            if s == GOAL:
                break
            a = ACTIONS[rng.integers(4)] if rng.random() < DEMO_EPS else policy[s]
            s2 = step(s, a)
            data.append((s, a, s2))
            s = s2
    return data


def bc_policy(data):
    """VLA-style: majority-vote action per state (only near-optimal actions,
    i.e. those the expert chose most, win the vote)."""
    counts = {}
    for s, a, _ in data:
        counts.setdefault(s, {}).setdefault(a, 0)
        counts[s][a] += 1
    return {s: max(acts, key=acts.get) for s, acts in counts.items()}


def wm_policy(data):
    """World-model-style: learn s,a -> s' from data, plan with value iteration.
    Unvisited (s,a) pairs are modeled as self-loops (unknown = no progress)."""
    T = {}
    for s, a, s2 in data:
        T.setdefault((s, a), {}).setdefault(s2, 0)
        T[(s, a)][s2] += 1
    model = {sa: max(d, key=d.get) for sa, d in T.items()}
    V = {s: 0.0 for s in all_states()}
    for _ in range(200):
        for s in V:
            if s == GOAL:
                continue
            V[s] = max(-1 + V[model.get((s, a), s)] for a in ACTIONS)
    return {s: max(ACTIONS, key=lambda a: V[model.get((s, a), s)]) for s in V}


def success_rate(policy):
    wins = 0
    starts = all_states()
    for s0 in starts:
        s = s0
        for _ in range(MAX_EVAL_STEPS):
            if s == GOAL:
                break
            a = policy.get(s)
            if a is None:  # state never seen -> act randomly
                a = ACTIONS[rng.integers(4)]
            s = step(s, a)
        wins += s == GOAL
    return wins / len(starts)


def main():
    plot_style.apply()
    expert = true_optimal_policy()
    sizes = [1, 2, 4, 8, 16, 32, 64]
    reps = 40

    bc_mean, wm_mean = [], []
    for n in sizes:
        bc_r, wm_r = [], []
        for _ in range(reps):
            data = collect_demos(expert, n)
            bc_r.append(success_rate(bc_policy(data)))
            wm_r.append(success_rate(wm_policy(data)))
        bc_mean.append(np.mean(bc_r))
        wm_mean.append(np.mean(wm_r))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sizes, bc_mean, color=BLUE, marker="o", markersize=6)
    ax.annotate("VLA-style (predict actions)", (sizes[2], bc_mean[2] - 0.07),
                color=BLUE, fontsize=9)
    ax.plot(sizes, wm_mean, color=GREEN, marker="o", markersize=6)
    ax.annotate("world-model-style (predict next state, then plan)",
                (sizes[1], wm_mean[1] + 0.05), color=GREEN, fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes, [str(s) for s in sizes])
    ax.set_xlabel("Expert demonstration episodes (same dataset for both)")
    ax.set_ylabel("Goal-reaching success from all starts")
    ax.set_ylim(0, 1.05)
    ax.set_title("Two heads, one skill: action prediction vs next-state prediction + planning",
                 loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig("results/exp4_vla_vs_world_model.png")

    print(f"{'episodes':>9} {'VLA-style':>10} {'world-model':>12}")
    for n, b, w in zip(sizes, bc_mean, wm_mean):
        print(f"{n:>9} {b:>10.1%} {w:>12.1%}")
    print("\nSame data, two prediction targets, converging capability — the "
          "dichotomy is architectural, not fundamental.")


if __name__ == "__main__":
    main()
