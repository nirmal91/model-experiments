"""Shared plumbing for experiment 2: tokenizer over the task charset, model
construction, checkpoint I/O, and greedy evaluation."""
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'experiment1_gpt_from_scratch'))
from model import GPT, GPTConfig          # noqa: E402  (experiment 1's transformer, reused)
from tokenizer import CharTokenizer       # noqa: E402

from tasks import CHARSET, EOS_CHAR, PROMPT_LEN, compute_reward, make_problem  # noqa: E402


def build_tokenizer() -> CharTokenizer:
    return CharTokenizer(''.join(CHARSET))


def build_model(tok) -> GPT:
    return GPT(GPTConfig(block_size=128, vocab_size=tok.vocab_size,
                         n_layer=4, n_head=4, n_embd=128, dropout=0.0))


def save_ckpt(model, path):
    torch.save({'model': model.state_dict(), 'config': model.config.__dict__}, path)


def load_ckpt(path) -> GPT:
    ckpt = torch.load(path, map_location='cpu')
    model = GPT(GPTConfig(**ckpt['config']))
    model.load_state_dict(ckpt['model'])
    return model


def save_history(history, path):
    with open(path, 'w') as f:
        json.dump(history, f, indent=2)


@torch.no_grad()
def evaluate(model, tok, n_problems=200, temperature=0.7, seed=123, greedy=True):
    """Sample completions for held-out problems; return accuracy/format rates
    and a few example transcripts. Greedy (temperature->argmax) by default,
    like pass@1 evals."""
    rng = random.Random(seed)
    model.eval()
    problems = [make_problem(rng) for _ in range(n_problems)]
    prompt_ids = torch.tensor([tok.encode(p) for p, _ in problems], dtype=torch.long)
    eos_id = tok.stoi[EOS_CHAR]

    idx = prompt_ids
    finished = torch.zeros(idx.size(0), dtype=torch.bool)
    for _ in range(80):
        logits, _ = model(idx[:, -model.config.block_size:])
        logits = logits[:, -1, :]
        if greedy:
            nxt = logits.argmax(dim=-1, keepdim=True)
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            nxt = torch.multinomial(probs, 1)
        nxt[finished] = eos_id
        idx = torch.cat((idx, nxt), dim=1)
        finished |= (nxt.squeeze(1) == eos_id)
        if finished.all():
            break

    n_correct = n_fmt = 0
    examples = []
    for i, (prompt, truth) in enumerate(problems):
        toks = idx[i, PROMPT_LEN:].tolist()
        if eos_id in toks:
            toks = toks[:toks.index(eos_id)]
        text = tok.decode(toks)
        _, c, f = compute_reward(text, truth)
        n_correct += c
        n_fmt += f
        if i < 5:
            examples.append(f"{prompt}{text}   [truth={truth} {'OK' if c else 'WRONG'}]")
    return {'accuracy': n_correct / n_problems, 'format': n_fmt / n_problems,
            'examples': examples}
