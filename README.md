# model-experiments

Two experiments, built to be read as much as run. Everything trains on a
plain 4-core CPU in minutes — no GPU required.

## [Experiment 1 — Build a GPT from scratch](experiment1_gpt_from_scratch/)

A next-token-predicting language model from first principles, following
Karpathy's *"Let's build GPT"* (+ the tokenizer video, + nanoGPT touches):
char tokenizer and a from-scratch BPE, a bigram baseline, then the full
decoder-only transformer — masked self-attention, multi-head, feed-forward,
residuals + pre-LayerNorm — trained on tiny Shakespeare.

## [Experiment 2 — DeepSeek-R1 at miniature scale](experiment2_deepseek_r1/)

Paper notes for *DeepSeek-R1* (arXiv:2501.12948) plus a working, faithful
small-scale implementation of its training recipe, reusing experiment 1's
transformer as the policy:

- **GRPO** from scratch (group-relative advantages, PPO-style clipping, k3 KL
  penalty to a frozen reference)
- rule-based **accuracy + format rewards** on `<think>…</think><answer>…</answer>`
- **R1-Zero**: pure RL from an unreliable pretrained base model
- **R1** stages 1–2: cold-start SFT → reasoning RL

The through-line: experiment 1 builds the *pretraining* stage (imitate the
data); experiment 2 shows what *reinforcement learning* adds (optimize an
outcome), on the exact same architecture.

## Setup

```bash
pip install -r requirements.txt
```

Then see each experiment's README for run instructions and results.
