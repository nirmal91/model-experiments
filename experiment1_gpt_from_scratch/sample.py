"""
Generate text from a trained checkpoint.

Usage:
    python sample.py                          # 500 tokens from empty context
    python sample.py --prompt "ROMEO:" -n 300
"""
import argparse
import pickle

import torch

from model import GPT, GPTConfig
from tokenizer import CharTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', default='gpt_shakespeare.pt')
parser.add_argument('--tokenizer', default='tokenizer.pkl')
parser.add_argument('--prompt', default='\n')
parser.add_argument('-n', '--num-tokens', type=int, default=500)
parser.add_argument('--temperature', type=float, default=0.8)
parser.add_argument('--top-k', type=int, default=40)
args = parser.parse_args()

ckpt = torch.load(args.ckpt, map_location='cpu')
model = GPT(GPTConfig(**ckpt['config']))
model.load_state_dict(ckpt['model'])
model.eval()

with open(args.tokenizer, 'rb') as f:
    d = pickle.load(f)
tok = CharTokenizer.__new__(CharTokenizer)
tok.stoi, tok.itos = d['stoi'], d['itos']
tok.vocab_size = len(tok.stoi)

idx = torch.tensor([tok.encode(args.prompt)], dtype=torch.long)
out = model.generate(idx, max_new_tokens=args.num_tokens,
                     temperature=args.temperature, top_k=args.top_k)
print(tok.decode(out[0].tolist()))
