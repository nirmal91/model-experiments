# The simplest machine that learns: a spam classifier

One file, no libraries, pure Python. It's a single-neuron model
(logistic regression) trained by a plain `for` loop.

```bash
python3 spam_classifier.py          # finished version
python3 spam_from_scratch.py        # workbook — build it yourself
```

## What kind of learning is this?

**Supervised learning** — specifically **binary classification** with
**logistic regression** (one neuron).

| Term | What it means here |
|------|--------------------|
| Supervised | Every training email comes with a **label** (`1` = spam, `0` = ham). The model learns by comparing its guess to that label. |
| Unsupervised | No labels — find structure on your own (clusters, topics). This project is **not** that. |
| Classification | Output is a category (spam vs not), not a continuous number like price. |
| Logistic regression | Weighted sum of features → sigmoid → probability in (0, 1). |

### Features: you chose them (and that's still supervised)

We **hard-coded three features** we care about:

1. count of known spammy words  
2. count of `!`  
3. whether the text has a link (`http` / `www`)

That does **not** make it unsupervised. Supervised vs unsupervised is about
**labels**, not about who invents the features.

- Classic supervised ML: **you** (or a pipeline) define features; the model
  only learns **weights**.
- Neural nets / representation learning: the model can learn useful internal
  features from raw input — still usually **supervised** if you have labels.
- Unsupervised: no labels at all.

So: hard-coded features + labeled emails = supervised learning with
**hand-engineered features**. Features do not have to "show up organically"
for learning to count as supervised.

## The mental model (learnings)

1. **Decide features** — what numbers describe an email?
2. **For each email**, compute those features (`get_features`).
3. **Guess** — `z = w·x + bias`, then `p = sigmoid(z)`.
4. **Compare** to the label → `error = label - predicted` (signed).
5. **Update** weights and bias from that error (this is the learning).
6. **Loss** is a dashboard only — e.g. average `|error|` or log loss. It
   tells you if you're making progress; it does **not** drive the update.
   The update uses `error` (and learning rate × feature).

Repeat over many epochs; weights converge; loss should fall.

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
| 1 | Go through many labeled examples | `TRAINING_EMAILS` |
| 2 | Find the 2–3 weights | 3 features → `weights = [w1, w2, w3]` + `bias` |
| 3 | Weighted sum → probability | `predict_probability()` + `sigmoid()` |
| 4 | Try a fresh input | `classify()` / `is_spam_or_not()` |
| 5 | A `for` loop of learning | `train()` / `build_model()` — watch loss fall |

## The whole learning rule (that's it)

For each example, over and over:

```
prediction = sigmoid(w·x + bias)     # what the model currently thinks
error      = label − prediction      # how wrong (signed)
weight[i] += rate · error · feature[i]   # nudge toward less wrong
bias      += rate · error
loss      += abs(error)              # dashboard only
```

(Equivalent form: `error = prediction − label` and subtract instead of add.)

Repeat enough times and the weights **converge**. Loss should drop from
~coin-flip territory toward something small (e.g. ~0.006 on this tiny set).

## Things worth trying

- Delete an example from `TRAINING_EMAILS` and watch the weights shift.
- Lower `learning_rate` (e.g. `0.001`) — convergence gets slow.
- Raise it too high (e.g. `2.0`) — the loss bounces instead of settling.
- Add a 4th feature (e.g. ALL-CAPS word count) — give it its own weight.
- Write your own fresh emails and see what it guesses.
