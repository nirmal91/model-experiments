"""
Stage 0: pretrain the "base model" for the R1 experiment.

Plays the role of DeepSeek-V3-Base: a next-token predictor over a corpus of
arithmetic documents in which ~25% of reasoning steps are corrupted and 15%
of documents skip the <think> section. The result is a model that fluently
speaks the task's language but is an *unreliable* reasoner — some latent
ability to answer correctly, far from consistent. That latent-but-unreliable
capability is exactly what R1-Zero's RL is supposed to incentivize.

Usage:  python pretrain_base.py [--iters 3000]
"""
import argparse
import time

import torch

from common import build_model, build_tokenizer, evaluate, save_ckpt
from tasks import make_pretrain_corpus

parser = argparse.ArgumentParser()
parser.add_argument('--iters', type=int, default=3000)
parser.add_argument('--smoke-test', action='store_true')
args = parser.parse_args()
max_iters = 20 if args.smoke_test else args.iters

batch_size = 48
block_size = 128
torch.manual_seed(1337)

tok = build_tokenizer()
corpus = make_pretrain_corpus(60_000, seed=0)
data = torch.tensor(tok.encode(corpus), dtype=torch.long)
print(f"corpus: {len(data):,} tokens, vocab_size={tok.vocab_size}")

model = build_model(tok)
print(f"model: {model.num_params():,} parameters")
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)


def get_batch():
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x, y


t0 = time.time()
for it in range(max_iters):
    x, y = get_batch()
    _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if it % 250 == 0 or it == max_iters - 1:
        print(f"step {it:5d} | loss {loss.item():.4f} | {time.time() - t0:.0f}s",
              flush=True)

save_ckpt(model, 'base_model.pt')
print("saved base_model.pt")

res = evaluate(model, tok, n_problems=64 if args.smoke_test else 200,
               greedy=False)  # sample like RL will, to measure the base rate
print(f"base model (sampled): accuracy={res['accuracy']:.2%} "
      f"format={res['format']:.2%}")
for e in res['examples']:
    print("  ", e)
