# Model 1 — the basic learning loop

This is the whole machine, independent of any one training run.

```
for each epoch:                          # do the homework set again
    for each labeled email:              # one row of training data
        guess  = model(email)            # produce a result (probability)
        error  = label - guess           # result vs expected
        nudge weights and bias           # so the next guess is less wrong
    report loss                          # average |error| that epoch (scoreboard)
```

Then stop. The fitted weights + bias **are the classifier**: new email in, spam / not spam out.

Files are numbered in learning order. **1a** and **1b** are the same model (two writeups). Next *model* is **2**. **3** is not built yet (train/test split).

```
1a_hand_features.py                # START — model 1 as you built it
1b_hand_features.py                # same model; longer demo, not a new step
2_bag_of_words_from_scratch.py     # NEXT — fill in the blanks
2_bag_of_words.py                  # model 2, finished
# 3_train_test.py                  # not yet — hold some emails out and score them
```

```bash
python3 1a_hand_features.py
python3 2_bag_of_words_from_scratch.py
python3 2_bag_of_words.py
```

---

## The four beats

**1. Iterate through epochs.**  
An epoch is one full pass over every training email. One pass is not enough; weights only move a little. Repeat (here: 200 times) until they settle. That settling is convergence.

**2. Each epoch, go through all training data and produce a result.**  
For each row: turn the email into three numbers, take `weighted sum + bias`, squash with a sigmoid. The result is `p(spam)` between 0 and 1.

**3. Measure result vs expected.**  
Expected is the **label** (`1` = spam, `0` = ham). The signed gap `error = label - guess` is what training uses. **Loss** is the epoch scoreboard — here, average `|error|`. Loss tells you the loop is working; it is not itself the knob-twist.

**4. Adjust weights and bias to reduce that gap.**  
In this file the nudge happens **after every email**, not once at the end of the epoch:

```
weight[i] += learning_rate × error × feature[i]
bias      += learning_rate × error
```

Same shared knobs for every row. A row with bigger features or a bigger error moves them more.

After training there is no more adjusting. `is_spam_or_not` only: features → probability → `True` if `> 0.5`.

---

## Names (Model 1)

| Jargon | Here |
|--------|------|
| Supervised learning | Every training email has a **label**. That is what “supervised” means — not “a human picked the features.” |
| Hand-engineered features | A human *did* pick the three measurements: spam-word count, `!` count, has-link. |
| Classifier | The finished function: email → spam / not spam. Training produces it. |
| Weights + bias | The knobs. Start at 0. After training they *are* the model. |
| Epoch | One lap through all 10 emails. |
| Learning rate | How hard each nudge twists the knobs (`0.1`). |
| Loss | Scoreboard for the epoch, not the update itself. |

Supervised vs unsupervised is about **labels**. Who invents the features is a separate choice (this model: you).

```
                w1
   x1 (spam words) ─┐
                w2  │
   x2 (! marks) ────┼──►  z = w1·x1 + w2·x2 + w3·x3 + bias  ──► sigmoid(z) ──► p(spam)
                w3  │
   x3 (has link) ───┘
```

---

## Notes that stay true after the run

- One set of weights for every row. Equal *vote count*, not equal *volume*: loud feature values and stubborn errors yank harder.
- Updating every row can be a lot of adjustment. If loss bounces, the learning rate is too high, or you would batch (one averaged update per epoch).
- With only 10 emails, one odd row is 10% of the data. Outliers speak loudly. Nothing here clips them.

---

## Things worth trying

- Delete an example from `TRAINING_EMAILS` and watch the weights shift.
- Lower `learning_rate` (e.g. `0.001`) — convergence gets slow.
- Raise it too high (e.g. `2.0`) — the loss bounces instead of settling.
- Add a 4th feature (e.g. ALL-CAPS word count) — give it its own weight.
- Write your own fresh emails and see what it guesses.

---

# Model 2: bag-of-words — `2_bag_of_words.py`

Same neuron. Same sigmoid. Same `for`-loop update. **One thing changes: the input.**

Model 1 made *you* pick 3 features and hand-write a `SPAM_WORDS` list. The model
only learned *how much* your 3 features mattered. Model 2 removes the hand-written
word list entirely: it builds a **vocabulary** from the training emails and gives
the neuron **one weight per word**. The model then **learns which words are spammy.**

| | Model 1 (features) | Model 2 (bag-of-words) |
|------|--------------------|------------------------|
| Features | 3, hand-designed | one per vocabulary word (~70 here) |
| Vocabulary | you wrote `SPAM_WORDS` | **built from the data** (`build_vocabulary`) |
| "free is spammy" | you asserted it | the model **learns** it (weight for `free` climbs) |
| Weights | 3 | as many as there are words |
| Learning rule | `w[i] -= rate·error·x[i]` | **exactly the same** |

### Bag-of-words in one sentence

Represent each email as a vector of word counts — for every word in the
vocabulary, how many times does it appear? `"free free win"` →
`{free: 2, win: 1, everything-else: 0}`. Order is discarded (it's a *bag*, not a
sequence). That vector *is* the feature list `x`; the neuron is unchanged.

### The payoff: you can read the model's mind

After training, sort the learned weights. Big positive = spam signal, big
negative = ham signal. On this dataset the model discovers, with no help:

```
SPAM:  +2.00 free   +1.30 http   +1.01 now   +1.00 money   +0.93 click   +0.91 win
HAM:   -1.03 thanks  -0.89 for    -0.74 to    -0.67 report  -0.67 quarterly
```

That list of spam words used to be *your* job (`SPAM_WORDS`). Now it's an
**output** of learning, not an input.

### The honest catch (what motivates model 3)

With ~70 weights and only 10 emails, training loss drops to ~0.003 — the model
can essentially **memorize** the training set. Fitting the training data
perfectly no longer proves it *generalizes*. The fix, and the next step, is a
**train/test split**: hold some emails out, train on the rest, and measure
accuracy on the ones the model never saw. Then TF-IDF weighting, then a hidden
layer.
