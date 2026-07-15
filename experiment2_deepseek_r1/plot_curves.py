"""Plot the RL training curves (reward, accuracy, format, response length)
for R1-Zero and the R1 pipeline. Produces curves.png.

Usage: python plot_curves.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load(path):
    if not Path(path).exists():
        return None
    with open(path) as f:
        return json.load(f)


runs = {'R1-Zero (pure RL from base)': load('r1_zero_history.json'),
        'R1 (cold-start SFT → RL)': load('r1_history.json')}
runs = {k: v for k, v in runs.items() if v}

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
panels = [('reward', 'Mean reward (max 1.5)'),
          ('accuracy', 'Accuracy (sampled rollouts)'),
          ('format', 'Format compliance'),
          ('resp_len', 'Mean response length (chars)')]
for ax, (key, title) in zip(axes.flat, panels):
    for name, hist in runs.items():
        ax.plot([h['step'] for h in hist], [h[key] for h in hist], label=name)
    ax.set_title(title)
    ax.set_xlabel('GRPO step')
    ax.grid(alpha=0.3)
axes.flat[0].legend(fontsize=9)
fig.suptitle('Mini DeepSeek-R1: GRPO training on 2-digit arithmetic')
fig.tight_layout()
fig.savefig('curves.png', dpi=120)
print('wrote curves.png')
