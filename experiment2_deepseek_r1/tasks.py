"""
The reasoning task + rule-based rewards for the mini DeepSeek-R1 experiment.

Task: 2-digit arithmetic. The prompt is e.g. "47+38=" and the model must
respond in DeepSeek-R1's format:

    <think>7+8=15,40+30=70,70+15=85</think><answer>85</answer>

Rewards are *rule-based*, exactly as in the paper (Sec 2.2.2) — no neural
reward model:

  - accuracy reward: 1.0 if the text inside <answer></answer> equals the
    ground-truth result
  - format reward: 0.5 if the completion matches
    <think>...</think><answer>...</answer>

The pretraining corpus (`make_pretrain_corpus`) deliberately contains
arithmetic errors (~45% of documents) and missing think-tags (~15%), so the
base model "speaks the language" of the task but is unreliable — the same
starting condition R1-Zero has: latent capability that RL must incentivize.
"""
import random
import re

PROMPT_LEN = 6           # every prompt is exactly "AB+CD=" / "AB-CD=" -> 6 chars
EOS_CHAR = '\n'

# fixed charset: digits, operators, tag characters, separators
CHARSET = sorted(set('0123456789+-=<>/thinkaswer,\n'))

FORMAT_RE = re.compile(r'^<think>([^<>]*)</think><answer>(\d{1,3})</answer>$')


def make_problem(rng: random.Random):
    """Return (prompt, ground_truth). Always 2-digit operands, non-negative
    result, so prompts are fixed-width and batch cleanly."""
    a, b = rng.randint(10, 99), rng.randint(10, 99)
    if rng.random() < 0.5:
        return f"{a}+{b}=", a + b
    a, b = max(a, b), min(a, b)
    return f"{a}-{b}=", a - b


def make_think(prompt: str, corrupt_prob: float, rng: random.Random):
    """Build a digit-decomposition reasoning trace for the problem, optionally
    corrupting intermediate results (the corpus's 'unreliable reasoner').
    Returns (think_text, final_total) — the final answer follows the (possibly
    wrong) reasoning, so errors are genuine reasoning errors, not typos."""
    a, op, b = int(prompt[0:2]), prompt[2], int(prompt[3:5])

    def maybe(x):
        return x + rng.choice([-10, -1, 1, 10]) if rng.random() < corrupt_prob else x

    if op == '+':
        ones = maybe(a % 10 + b % 10)
        tens = maybe(a // 10 * 10 + b // 10 * 10)
        total = tens + ones
        think = f"{a % 10}+{b % 10}={ones},{a // 10 * 10}+{b // 10 * 10}={tens},{tens}+{ones}={total}"
    else:
        tens = maybe(a // 10 * 10 - b // 10 * 10)
        ones = maybe(a % 10 - b % 10)
        total = tens + ones
        think = f"{a // 10 * 10}-{b // 10 * 10}={tens},{a % 10}-{b % 10}={ones},{tens}{ones:+d}={total}"
    return think, max(0, total)


def make_document(rng: random.Random, corrupt_prob=0.25, no_think_prob=0.15):
    """One pretraining document: prompt + completion + newline."""
    prompt, truth = make_problem(rng)
    if rng.random() < no_think_prob:
        ans = truth if rng.random() > corrupt_prob else max(0, truth + rng.choice([-10, -1, 1, 10]))
        return f"{prompt}<answer>{ans}</answer>{EOS_CHAR}"
    think, total = make_think(prompt, corrupt_prob, rng)
    return f"{prompt}<think>{think}</think><answer>{total}</answer>{EOS_CHAR}"


def make_pretrain_corpus(n_docs: int, seed=0) -> str:
    rng = random.Random(seed)
    return ''.join(make_document(rng) for _ in range(n_docs))


def make_sft_example(rng: random.Random):
    """A perfect cold-start example: correct reasoning, correct format.
    Returns (prompt, completion)."""
    prompt, _ = make_problem(rng)
    think, total = make_think(prompt, corrupt_prob=0.0, rng=rng)
    return prompt, f"<think>{think}</think><answer>{total}</answer>{EOS_CHAR}"


# ---------------- rule-based rewards (paper Sec 2.2.2) ----------------

def compute_reward(completion: str, ground_truth: int):
    """Returns (total_reward, is_correct, is_well_formatted).
    completion is the raw sampled text after the prompt, up to (not
    including) the EOS newline."""
    m = FORMAT_RE.match(completion.strip())
    fmt = m is not None
    correct = fmt and int(m.group(2)) == ground_truth
    return (1.0 * correct) + (0.5 * fmt), correct, fmt


if __name__ == '__main__':
    rng = random.Random(42)
    print("charset:", ''.join(CHARSET).replace('\n', '\\n'))
    corpus = make_pretrain_corpus(5, seed=1)
    print("--- sample corpus docs ---")
    print(corpus, end='')
    assert set(corpus) <= set(CHARSET), "corpus uses chars outside CHARSET"

    # reward sanity checks
    assert compute_reward("<think>x</think><answer>85</answer>", 85) == (1.5, True, True)
    assert compute_reward("<think>x</think><answer>84</answer>", 85) == (0.5, False, True)
    assert compute_reward("<answer>85</answer>", 85) == (0.0, False, False)
    assert compute_reward("garbage", 85) == (0.0, False, False)
    p, c = make_sft_example(rng)
    print("--- sft example ---")
    print(p + c, end='')
    assert compute_reward(c.strip(), eval(p[:-1]))[1]
    print("all reward checks passed")
