# The simplest machine that learns: a spam classifier

One file, no libraries, pure Python. It's a single-neuron model
(logistic regression) trained by a plain `for` loop. Run it:

```bash
python3 spam_classifier.py
```

## What it actually is

Not a deep neural network — it's **one neuron**. It takes a few numbers,
multiplies each by a **weight**, adds them up, and squashes the total into a
**probability** between 0 and 1. If that probability is ≥ 0.5, it says "spam".

```
                w1
   x1 (spam words) ─┐
                w2  │
   x2 (! marks) ────┼──►  z = w1·x1 + w2·x2 + w3·x3 + bias  ──► sigmoid(z) ──► p(spam)
                w3  │
   x3 (has link) ───┘
```

## The 5 steps, mapped to the code

| Step | Idea | Where |
|------|------|-------|
| 1 | Go through many examples | `TRAINING_EMAILS` |
| 2 | Find the 2–3 weights | 3 features → `weights = [w1, w2, w3]` + `bias` |
| 3 | Weighted sum → probability | `predict_probability()` + `sigmoid()` |
| 4 | Try a fresh input | `classify()` on `fresh_emails` |
| 5 | A `for` loop of learning | `train()` — watch the weights converge |

## The whole learning rule (that's it)

For each example, over and over:

```
prediction = sigmoid(w·x + bias)     # what the model currently thinks
error      = prediction − truth      # how wrong it was
weight     = weight − rate · error · feature   # nudge toward less wrong
```

Repeat enough times and the weights stop moving — they **converge**. In the
run you'll see them climb from `0` and settle (e.g. `w1 ≈ 1.76`), while the
average loss falls from ~0.47 to ~0.006.

## Things worth trying

- Delete an example from `TRAINING_EMAILS` and watch the weights shift.
- Lower `learning_rate` (e.g. `0.001`) — convergence gets slow.
- Raise it too high (e.g. `2.0`) — the loss bounces instead of settling.
- Add a 4th feature (e.g. ALL-CAPS word count) — give it its own weight.
- Write your own `fresh_emails` and see what it guesses.
