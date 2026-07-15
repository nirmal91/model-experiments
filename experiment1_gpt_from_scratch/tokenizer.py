"""
Tokenizers, from scratch.

Two tokenizers live here:

1. CharTokenizer — the character-level tokenizer used in Karpathy's
   "Let's build GPT" video. Trivially simple: every unique character in the
   corpus becomes one token. Great for small experiments (vocab ~65 for
   Shakespeare) because the model spends no capacity learning spelling of a
   huge vocab — but sequences are long since 1 char = 1 token.

2. BPETokenizer — a minimal byte-pair-encoding tokenizer in the spirit of
   Karpathy's "Let's build the GPT Tokenizer" video (minbpe). Starts from
   raw bytes (vocab 256) and repeatedly merges the most frequent adjacent
   pair into a new token. This is (a simplified version of) what GPT-2/3/4
   actually use.

Experiment 1 trains with CharTokenizer; BPETokenizer is here to make the
"tokenization is its own model" lesson concrete. Run this file directly for
a quick demo/self-test of both.
"""


class CharTokenizer:
    """Character-level tokenizer: one token per unique character."""

    def __init__(self, text: str):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return ''.join(self.itos[i] for i in ids)


class BPETokenizer:
    """Minimal byte-level BPE, as in GPT-2 (without the regex split pattern).

    train(): find the most common adjacent byte/token pair, mint a new token
    for it, repeat until vocab_size is reached. encode()/decode() replay the
    learned merges in order.
    """

    def __init__(self):
        self.merges = {}          # (int, int) -> int
        self.vocab = {i: bytes([i]) for i in range(256)}

    @staticmethod
    def _pair_counts(ids: list[int]) -> dict:
        counts = {}
        for a, b in zip(ids, ids[1:]):
            counts[(a, b)] = counts.get((a, b), 0) + 1
        return counts

    @staticmethod
    def _merge(ids: list[int], pair: tuple, new_id: int) -> list[int]:
        out, i = [], 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        assert vocab_size >= 256
        ids = list(text.encode('utf-8'))
        for i in range(vocab_size - 256):
            counts = self._pair_counts(ids)
            if not counts:
                break
            pair = max(counts, key=counts.get)
            new_id = 256 + i
            ids = self._merge(ids, pair, new_id)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose and i % 50 == 0:
                print(f"merge {i}: {pair} -> {new_id} ({self.vocab[new_id]!r})")

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode('utf-8'))
        while len(ids) >= 2:
            counts = self._pair_counts(ids)
            # merge the pair that was learned earliest (lowest new token id)
            pair = min(counts, key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break
            ids = self._merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids: list[int]) -> str:
        return b''.join(self.vocab[i] for i in ids).decode('utf-8', errors='replace')


if __name__ == '__main__':
    with open('data/tinyshakespeare.txt', 'r') as f:
        text = f.read()

    # char tokenizer round trip
    ct = CharTokenizer(text)
    s = "To be, or not to be"
    assert ct.decode(ct.encode(s)) == s
    print(f"CharTokenizer: vocab_size={ct.vocab_size}, "
          f"{len(s)} chars -> {len(ct.encode(s))} tokens")

    # BPE tokenizer round trip + compression demo
    bpe = BPETokenizer()
    bpe.train(text[:100_000], vocab_size=512)
    enc = bpe.encode(s)
    assert bpe.decode(enc) == s
    ratio = len(s.encode('utf-8')) / len(enc)
    print(f"BPETokenizer:  vocab_size=512, {len(s)} chars -> {len(enc)} tokens "
          f"(compression {ratio:.2f}x)")
    print("sample merges:", [bpe.vocab[256 + i] for i in range(10)])
