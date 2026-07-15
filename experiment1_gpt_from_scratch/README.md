# Experiment 1 — Build a GPT from scratch

A decoder-only transformer language model built from first principles,
following Andrej Karpathy's [*Let's build GPT: from scratch, in code, spelled
out*](https://www.youtube.com/watch?v=kCc8FmEb1nY) (and nanoGPT), trained to
predict the next character of tiny Shakespeare on a plain CPU.

## The build progression

The video's arc is reproduced in this directory, one concept at a time:

| Step | Where | Idea |
|---|---|---|
| 0 | `tokenizer.py` | Text → integers. Char-level (vocab 65) for training; a from-scratch byte-level **BPE** (à la *Let's build the GPT Tokenizer*) included for comparison |
| 1 | `bigram.py` | Baseline: predict next char from a lookup table of the current char only. Loss ~2.5, output is noise with Shakespeare-ish letter statistics |
| 2 | `model.py` — `CausalSelfAttention` | **Self-attention**: tokens communicate. Queries/keys give affinities, a lower-triangular mask hides the future, values are averaged. The masked-matmul trick trains all T positions in parallel |
| 3 | `model.py` — multi-head | Several attention channels in parallel, concatenated |
| 4 | `model.py` — `MLP` | Communicate (attention) then compute (per-token feed-forward, 4× expansion) |
| 5 | `model.py` — `Block` | **Residual connections + pre-LayerNorm** so deep stacks optimize; GPT-2-style init scaling of residual projections |
| 6 | `train.py` | AdamW, cosine LR with warmup, grad clipping, train/val loss estimation, dropout |
| 7 | `sample.py` / `GPT.generate` | Autoregressive sampling with temperature and top-k |

## Model

~0.81M parameters: 4 layers, 4 heads, 128-dim embeddings, 128-char context,
character-level vocab of 65, weight tying between token embedding and output
head.

## Run it

```bash
python bigram.py            # the baseline (~1 min)
python train.py             # the real thing (~15 min on a 4-core CPU)
python sample.py --prompt "ROMEO:" -n 400
python tokenizer.py         # char vs BPE demo/self-test
```

## Results (this repo's actual run, 3000 iters, CPU)

<!-- RESULTS -->

Losses to beat: bigram baseline converges around **2.45** val loss; random
guessing over 65 chars is ln(65) ≈ **4.17**.

Full logs in `train_log.txt`, loss history in `train_history.json`, longer
sample in `sample.txt`.

## What the model is (and isn't)

This is a *pretrained next-token predictor* — the "GPT" stage of the modern
pipeline, before any fine-tuning or RL. It continues text; it doesn't follow
instructions. Experiment 2 takes exactly this architecture and shows what
reinforcement learning adds on top.
