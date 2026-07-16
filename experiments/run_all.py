"""Run all experiments and write figures to results/."""

import os
import runpy
import sys

os.makedirs("results", exist_ok=True)
for name in ("exp1_perception_overhang", "exp2_action_space_dimensionality",
             "exp3_delta_actions_vs_primitives", "exp4_vla_vs_world_model"):
    print(f"\n=== {name} ===")
    runpy.run_module(name, run_name="__main__")
