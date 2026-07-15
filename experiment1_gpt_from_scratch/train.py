"""
Train the from-scratch GPT (model.py) on tiny Shakespeare, character-level.

This is the training loop from the "Let's build GPT" video with a couple of
nanoGPT niceties (cosine LR schedule with warmup, grad clipping, checkpoint
saving). Sized to train in ~10-15 minutes on a 4-core CPU.

Usage:
    python train.py                 # full run
    python train.py --smoke-test    # 20 iters, sanity check
"""
import argparse
import json
import math
import pickle
import time

import torch

from model import GPT, GPTConfig
from tokenizer import CharTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--smoke-test', action='store_true')
parser.add_argument('--max-iters', type=int, default=3000)
args = parser.parse_args()

# ---------------- hyperparameters ----------------
batch_size = 32
block_size = 128
max_iters = 20 if args.smoke_test else args.max_iters
eval_interval = 5 if args.smoke_test else 250
eval_iters = 2 if args.smoke_test else 100
learning_rate = 1e-3
warmup_iters = 100
min_lr = 1e-4
grad_clip = 1.0
device = 'cpu'
torch.manual_seed(1337)

# ---------------- data ----------------
with open('data/tinyshakespeare.txt', 'r') as f:
    text = f.read()
tok = CharTokenizer(text)
data = torch.tensor(tok.encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
print(f"dataset: {len(data):,} tokens, vocab_size={tok.vocab_size}")


def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + 1 + block_size] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters
    decay = (it - warmup_iters) / max(1, max_iters - warmup_iters)
    return min_lr + 0.5 * (learning_rate - min_lr) * (1 + math.cos(math.pi * decay))


# ---------------- model ----------------
config = GPTConfig(
    block_size=block_size, vocab_size=tok.vocab_size,
    n_layer=4, n_head=4, n_embd=128, dropout=0.1,
)
model = GPT(config)
print(f"model: {model.num_params():,} parameters "
      f"({config.n_layer} layers, {config.n_head} heads, {config.n_embd} dim)")
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                              betas=(0.9, 0.99), weight_decay=0.1)

# ---------------- training loop ----------------
history = []
t0 = time.time()
for it in range(max_iters):
    lr = get_lr(it)
    for g in optimizer.param_groups:
        g['lr'] = lr

    if it % eval_interval == 0 or it == max_iters - 1:
        losses = estimate_loss(model)
        dt = time.time() - t0
        print(f"step {it:5d} | train {losses['train']:.4f} | "
              f"val {losses['val']:.4f} | lr {lr:.2e} | {dt:.0f}s", flush=True)
        history.append({'iter': it, **losses, 'time_s': round(dt, 1)})

    x, y = get_batch('train')
    _, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

# ---------------- save ----------------
suffix = '_smoke' if args.smoke_test else ''
torch.save({'model': model.state_dict(), 'config': config.__dict__},
           f'gpt_shakespeare{suffix}.pt')
with open(f'tokenizer{suffix}.pkl', 'wb') as f:
    pickle.dump({'stoi': tok.stoi, 'itos': tok.itos}, f)
with open(f'train_history{suffix}.json', 'w') as f:
    json.dump(history, f, indent=2)
print(f"saved gpt_shakespeare{suffix}.pt")

# ---------------- sample ----------------
model.eval()
context = torch.zeros((1, 1), dtype=torch.long)
sample = tok.decode(model.generate(context, max_new_tokens=500, temperature=0.8,
                                   top_k=40)[0].tolist())
print('--- sample ---')
print(sample)
with open(f'sample{suffix}.txt', 'w') as f:
    f.write(sample)
