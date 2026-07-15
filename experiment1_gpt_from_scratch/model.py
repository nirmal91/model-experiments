"""
A GPT (decoder-only transformer), from scratch, spelled out.

This follows the progression of Karpathy's "Let's build GPT" video / nanoGPT:

  bigram (bigram.py: next token from a lookup table — no context)
    -> self-attention (tokens *communicate*: each position takes a weighted
       average of information from prior positions, weights computed from
       query/key dot products)
    -> multi-head attention (several attention "channels" in parallel)
    -> feed-forward layer (per-token computation after communication)
    -> residual connections + LayerNorm (so deep stacks optimize well;
       pre-norm variant as in GPT-2)
    -> dropout (regularization)

The mathematical trick at the heart of it: masked matrix multiply.
A lower-triangular mask on the (T, T) attention matrix means position t can
only attend to positions <= t, so the model can be trained on all T
next-token predictions in a sequence at once, in parallel.

The same GPT class is reused by Experiment 2 (DeepSeek R1) as the policy
network — a language model is a language model; only the training signal
changes.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 128    # maximum context length
    vocab_size: int = 65
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = False       # GPT-2 uses biases; False is a bit faster/cleaner


class CausalSelfAttention(nn.Module):
    """Multi-head masked self-attention.

    Every token emits a query ("what am I looking for?") and a key ("what do
    I contain?"). Affinity between tokens = q @ k^T, scaled by 1/sqrt(head
    dim) so softmax stays diffuse at init. The causal mask forbids looking at
    the future. The output is an affinity-weighted sum of value vectors.
    All heads are computed in one batched matmul.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # q, k, v projections for all heads, in a single linear layer
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # causal mask: (1, 1, T, T) lower-triangular
        self.register_buffer(
            'mask',
            torch.tril(torch.ones(config.block_size, config.block_size))
                 .view(1, 1, config.block_size, config.block_size),
        )

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim)
        hd = C // self.n_head
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)

        # attention scores: (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hd))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v                                   # (B, nh, T, hd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """Position-wise feed-forward network: after tokens have communicated via
    attention, each token 'thinks' on its own. 4x expansion as in GPT-2."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


class Block(nn.Module):
    """Transformer block: communicate (attention) then compute (MLP).

    Pre-norm residual form: x = x + f(LayerNorm(x)). The residual stream is a
    'gradient superhighway' — gradients flow unimpeded from the loss to every
    layer, which is what lets us stack many blocks and still optimize.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),   # token embeddings
            wpe=nn.Embedding(config.block_size, config.n_embd),   # position embeddings
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight tying: input embedding and output projection share weights
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # GPT-2 style: scale residual projections by 1/sqrt(2*n_layer) so the
        # residual stream's variance doesn't grow with depth
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size
        pos = torch.arange(T, device=idx.device)

        tok_emb = self.transformer.wte(idx)      # (B, T, C) what the token is
        pos_emb = self.transformer.wpe(pos)      # (T, C)    where the token is
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)                 # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # targets of -1 are masked out of the loss (used by SFT in exp 2)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None,
                 stop_token=None):
        """Autoregressive sampling: feed the sequence in, sample the next
        token from the softmax, append, repeat."""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size \
                else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if stop_token is not None and (idx_next == stop_token).all():
                break
        return idx
