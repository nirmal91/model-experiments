"""
Mini DeepSeek-R1: the first two stages of the paper's 4-stage pipeline.

  Stage 1 — cold start: SFT the base model on a small set of *perfect*
            reasoning examples (correct trace, correct format). In the paper
            this is thousands of human-cleaned long CoTs; here it's 512
            synthetic ones. Loss is masked to completion tokens only.
  Stage 2 — reasoning RL: same GRPO recipe as R1-Zero, but starting from the
            cold-started checkpoint, with the SFT model as KL reference.

(Stages 3-4 — rejection-sampling SFT and general-scenario RL — need general
non-reasoning data and a preference reward model; out of scope here.)

    base_model.pt --SFT--> r1_coldstart.pt --GRPO--> r1.pt

Usage:  python train_r1.py [--sft-epochs 3] [--steps 150]
"""
import argparse
import copy
import random

import torch

from common import build_tokenizer, evaluate, load_ckpt, save_ckpt, save_history
from grpo import GRPOConfig, GRPOTrainer
from tasks import EOS_CHAR, PROMPT_LEN, compute_reward, make_problem, make_sft_example

parser = argparse.ArgumentParser()
parser.add_argument('--sft-examples', type=int, default=512)
parser.add_argument('--sft-epochs', type=int, default=3)
parser.add_argument('--steps', type=int, default=150)
parser.add_argument('--smoke-test', action='store_true')
args = parser.parse_args()

torch.manual_seed(7)
tok = build_tokenizer()
model = load_ckpt('base_model.pt')

# ---------------- stage 1: cold-start SFT ----------------
rng = random.Random(99)
examples = [make_sft_example(rng) for _ in range(args.sft_examples)]

rows, targets = [], []
maxlen = max(len(p) + len(c) for p, c in examples)
pad = tok.stoi[EOS_CHAR]
for p, c in examples:
    ids = tok.encode(p + c)
    t = [-1] * (PROMPT_LEN - 1) + ids[PROMPT_LEN:]  # predict completion only
    ids = ids + [pad] * (maxlen - len(ids))
    t = t + [-1] * (maxlen - 1 - len(t))
    rows.append(ids)
    targets.append(t)
X = torch.tensor(rows, dtype=torch.long)
Y = torch.tensor(targets, dtype=torch.long)

opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
bs = 32
epochs = 1 if args.smoke_test else args.sft_epochs
print(f"cold-start SFT: {len(examples)} examples, {epochs} epochs")
for epoch in range(epochs):
    perm = torch.randperm(len(examples))
    tot = 0.0
    for i in range(0, len(examples), bs):
        j = perm[i:i + bs]
        _, loss = model(X[j, :-1], Y[j])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        tot += loss.item() * len(j)
    print(f"  epoch {epoch}: sft loss {tot / len(examples):.4f}", flush=True)

save_ckpt(model, 'r1_coldstart.pt')
mid = evaluate(model, tok, n_problems=32 if args.smoke_test else 200)
print(f"after cold start (greedy): accuracy={mid['accuracy']:.2%} "
      f"format={mid['format']:.2%}")

# ---------------- stage 2: reasoning RL (GRPO) ----------------
ref = copy.deepcopy(model)  # KL reference = the SFT checkpoint
cfg = GRPOConfig(total_steps=3 if args.smoke_test else args.steps,
                 group_size=8, prompts_per_step=8, kl_beta=0.02,
                 inner_epochs=2, lr=3e-4, seed=1)
history = GRPOTrainer(model, ref, tok, make_problem, compute_reward, cfg,
                      eos_char=EOS_CHAR).train()

save_ckpt(model, 'r1.pt')
save_history(history, 'r1_history.json')

post = evaluate(model, tok, n_problems=32 if args.smoke_test else 200)
print(f"after RL (greedy): accuracy={post['accuracy']:.2%} format={post['format']:.2%}")
print('--- final samples ---')
for e in post['examples']:
    print("  ", e)
