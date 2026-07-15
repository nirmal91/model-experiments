# DeepSeek-R1 — Paper Reading Notes

**Paper:** *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via
Reinforcement Learning* (DeepSeek-AI, Jan 2025, arXiv:2501.12948)

## The core claim

Reasoning capability can be **incentivized with pure reinforcement learning** —
no supervised fine-tuning on reasoning traces required. Given only a reward for
*getting the right answer* (and for following an output format), a base model
learns by itself to produce long chains of thought, to self-verify, to reflect,
and to backtrack ("wait, let me reconsider..."). Behaviors nobody wrote
demonstrations for **emerge** because they help the model earn reward.

## The two models

### 1. DeepSeek-R1-Zero — pure RL, no SFT

- Start from the **base model** (DeepSeek-V3-Base, 671B MoE) — no instruction
  tuning, no SFT whatsoever.
- Train with **GRPO** (Group Relative Policy Optimization) against
  **rule-based rewards** only:
  - **Accuracy reward** — deterministic check that the final answer is correct
    (math answers checked against ground truth, code run against test cases).
  - **Format reward** — the model must put its reasoning between
    `<think>` ... `</think>` tags and the answer between `<answer>` ...
    `</answer>` tags.
  - Deliberately **no neural reward model** — the paper found reward models
    suffer reward hacking at scale, and retraining them costs compute.
- A minimal prompt template just tells the model to think first, then answer —
  content of the reasoning is entirely unconstrained ("we let the model figure
  out the best way to think").

**What happened:**
- AIME 2024 pass@1 climbed from **15.6% → 71.0%** over RL training (86.7% with
  majority voting) — comparable to OpenAI o1-0912.
- **Response length grew steadily** through training — the model *learned* to
  spend more test-time compute thinking, without being told to.
- The **"aha moment"**: mid-training, the model spontaneously starts re-evaluating
  its own steps — the paper shows a sample where it interrupts itself with
  *"Wait, wait. Wait. That's an aha moment I can flag here."* and re-derives
  the solution. Reflection and alternative-exploration emerged, not programmed.
- **Drawbacks:** poor readability, language mixing (Chinese/English swapped
  mid-thought). Great reasoner, unpleasant assistant. Hence R1 proper.

### 2. DeepSeek-R1 — the 4-stage pipeline

Fixes R1-Zero's readability problems and generalizes beyond reasoning:

1. **Cold start (SFT):** collect a few **thousand** long, human-readable CoT
   examples (few-shot prompting, R1-Zero outputs cleaned by human annotators,
   readable format with a summary at the end) and fine-tune V3-Base on them.
   Purpose: readable output pattern + faster RL convergence, not capability.
2. **Reasoning-oriented RL:** same GRPO recipe as R1-Zero on the cold-started
   model, plus a **language-consistency reward** (fraction of target-language
   words in the CoT) to stop language mixing — costs a little accuracy, wins
   on human preference.
3. **Rejection sampling + SFT:** when reasoning RL converges, sample from the
   checkpoint and keep only correct/readable outputs → **~600k reasoning
   samples**; add **~200k non-reasoning samples** (writing, factual QA,
   translation, from the V3 pipeline) → retrain V3-Base on the ~800k for 2
   epochs. This is where general helpfulness comes back in.
4. **RL for all scenarios:** a second RL stage mixing rule-based rewards (for
   reasoning prompts) with **reward models** for helpfulness/harmlessness on
   general prompts.

Result: R1 matches OpenAI o1-1217 on reasoning benchmarks (AIME 79.8%,
MATH-500 97.3%, Codeforces 96.3 percentile).

### 3. Distillation

Fine-tune small dense models (Qwen2.5 1.5B–32B, Llama-3 8B/70B) on the 800k
R1-generated samples — **SFT only, no RL**. Key finding: **distilling from a
big RL-trained teacher beats running RL directly on the small model**
(RL on Qwen-32B ≈ QwQ-32B-Preview; distilled Qwen-32B far better). The
reasoning patterns discovered by large-scale RL are the valuable artifact —
small models can copy them cheaply but struggle to discover them.

## GRPO — the algorithm (from DeepSeekMath, used throughout R1)

PPO needs a critic (value network) as large as the policy to estimate
advantages. GRPO **drops the critic**: sample a *group* of G outputs
{o₁...o_G} per question q from the old policy, and use the group's own reward
statistics as the baseline:

```
Â_i = (r_i − mean(r₁..r_G)) / std(r₁..r_G)
```

Objective (clipped surrogate like PPO, plus explicit KL to a reference policy):

```
J(θ) = E[ 1/G Σ_i ( min( ρ_i·Â_i,  clip(ρ_i, 1−ε, 1+ε)·Â_i ) − β·D_KL(π_θ ‖ π_ref) ) ]
where ρ_i = π_θ(o_i|q) / π_θ_old(o_i|q)
```

The KL term uses the low-variance, always-positive **k3 estimator** rather
than appearing inside the reward:

```
D_KL(π_θ ‖ π_ref) ≈ π_ref(o|q)/π_θ(o|q) − log( π_ref(o|q)/π_θ(o|q) ) − 1
```

Intuition: within a group, completions that scored above the group average get
pushed up, those below get pushed down — the group mean plays the role of the
value function. Cheap, simple, and stable enough to run at 671B scale.

## Negative results the paper reports (worth remembering)

- **Process Reward Models (PRMs):** hard to define step granularity, hard to
  label intermediate-step correctness, and the neural reward model invites
  reward hacking. Not worth it at scale.
- **MCTS (AlphaGo-style search):** token generation's search space is
  exponentially larger than a board game's; a good value model to guide search
  is hard to train; scaling search hit a wall.

Both failures point the same direction: **simple outcome-based rewards +
large-scale RL** beat cleverer, more structured supervision.

## What we reproduce in this experiment (and what we don't)

| Paper | This repo |
|---|---|
| DeepSeek-V3-Base, 671B MoE | ~0.9M-param char-level GPT (from experiment 1's `model.py`) |
| Math/code/STEM questions | 2-digit arithmetic (`47+38=`) |
| Accuracy + format rewards, rule-based | Same, literally (`tasks.py`) |
| GRPO with clip + k3 KL penalty | Same math (`grpo.py`) |
| R1-Zero: pure RL from base | `train_r1_zero.py` |
| R1: cold-start SFT → reasoning RL | `train_r1.py` (stages 1–2 of 4) |
| Stages 3–4 (rejection sampling, general RL), distillation | Out of scope on a laptop-class CPU |

The point is not to reproduce the numbers — it's that the *mechanism* is
scale-free enough to watch happen: a base model that's right ~half the time,
trained only on "was the answer correct + was the format right", climbs to
near-perfect accuracy, and the group-relative advantage does all the work.
