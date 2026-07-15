"""
Step 0 of building GPT from scratch: the bigram language model.

This is the starting point of Karpathy's "Let's build GPT: from scratch,
in code, spelled out" video. The bigram model predicts the next character
from ONLY the current character, via a single embedding-table lookup:

    logits[next_char] = table[current_char]

There is no context beyond one token — which is exactly why we then invent
self-attention (see model.py). Run this first to get a baseline loss, then
compare against the full GPT.

Usage:
    python bigram.py
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

from tokenizer import CharTokenizer

# hyperparameters
batch_size = 32
block_size = 8       # context length (irrelevant for bigram, used for batching)
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
eval_iters = 200
device = 'cpu'
torch.manual_seed(1337)

# data
with open('data/tinyshakespeare.txt', 'r') as f:
    text = f.read()
tok = CharTokenizer(text)
data = torch.tensor(tok.encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]


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
        out[split] = losses.mean()
    model.train()
    return out


class BigramLanguageModel(nn.Module):
    """Each token directly reads off the logits for the next token from a
    lookup table. That's the entire model."""

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]                     # last time step
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


if __name__ == '__main__':
    model = BigramLanguageModel(tok.vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for it in range(max_iters):
        if it % eval_interval == 0 or it == max_iters - 1:
            losses = estimate_loss(model)
            print(f"step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        xb, yb = get_batch('train')
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    context = torch.zeros((1, 1), dtype=torch.long)
    print('--- bigram sample ---')
    print(tok.decode(model.generate(context, max_new_tokens=300)[0].tolist()))
