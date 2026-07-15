"""
Mini DeepSeek-R1-Zero: pure RL on the base model. No SFT, no human data —
only GRPO against rule-based accuracy + format rewards.

    base_model.pt --GRPO--> r1_zero.pt

Logs reward/accuracy/format/response-length curves (r1_zero_history.json)
and a transcript of sampled completions over training (r1_zero_samples.txt)
so you can watch the behavior change.

Usage:  python train_r1_zero.py [--steps 150]
"""
import argparse
import copy
import random

import torch

from common import build_tokenizer, evaluate, load_ckpt, save_ckpt, save_history
from grpo import GRPOConfig, GRPOTrainer
from tasks import EOS_CHAR, compute_reward, make_problem

parser = argparse.ArgumentParser()
parser.add_argument('--steps', type=int, default=150)
parser.add_argument('--smoke-test', action='store_true')
args = parser.parse_args()

torch.manual_seed(7)
tok = build_tokenizer()
policy = load_ckpt('base_model.pt')
ref = copy.deepcopy(policy)   # frozen reference for the KL penalty

pre = evaluate(policy, tok, n_problems=32 if args.smoke_test else 200)
print(f"before RL (greedy): accuracy={pre['accuracy']:.2%} format={pre['format']:.2%}")

cfg = GRPOConfig(total_steps=3 if args.smoke_test else args.steps,
                 group_size=8, prompts_per_step=8, kl_beta=0.02,
                 inner_epochs=2, lr=3e-4)

sample_file = open('r1_zero_samples.txt', 'w')


def log_samples(trainer, step):
    """Dump a couple of raw rollouts so the emergent behavior is visible."""
    rng = random.Random(1000 + step)
    prompt, truth = make_problem(rng)
    idx = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    out = trainer.policy.generate(idx, 80, temperature=0.8,
                                  stop_token=tok.stoi[EOS_CHAR])
    text = tok.decode(out[0, len(prompt):].tolist()).strip()
    r, c, f = compute_reward(text, truth)
    sample_file.write(f"[step {step}] {prompt}{text}   "
                      f"(truth={truth}, reward={r}, correct={c})\n")
    sample_file.flush()


history = GRPOTrainer(policy, ref, tok, make_problem, compute_reward, cfg,
                      eos_char=EOS_CHAR).train(sample_logger=log_samples)
sample_file.close()

save_ckpt(policy, 'r1_zero.pt')
save_history(history, 'r1_zero_history.json')

post = evaluate(policy, tok, n_problems=32 if args.smoke_test else 200)
print(f"after RL (greedy):  accuracy={post['accuracy']:.2%} format={post['format']:.2%}")
print('--- final samples ---')
for e in post['examples']:
    print("  ", e)
