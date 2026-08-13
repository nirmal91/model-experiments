"""
WORKBOOK: build the bag-of-words spam classifier yourself (Model 2).
====================================================================

Same idea as spam_from_scratch.py, one level up. In Model 1 you hand-wrote a
SPAM_WORDS list. Here there is NO such list: you build a VOCABULARY from the
training emails and give the model ONE weight per word, so it learns which
words are spammy on its own.

Fill in every spot marked  # >>> YOUR CODE HERE.
Each blank has a hint telling you the one or two lines to write.

Run it any time:   python3 spam_bow_from_scratch.py
While the blanks are empty it will FAIL the self-check -- that's expected.
Keep filling them in and re-running. When it prints "ALL CHECKS PASSED",
you've rebuilt Model 2 from scratch.

Stuck? Peek at the finished version in spam_classifier_bow.py.
"""

import math
import re


# ---------------------------------------------------------------------------
# GIVEN: the examples. label = 1 means SPAM, 0 means NOT spam (ham).
# ---------------------------------------------------------------------------
TRAINING_EMAILS = [
    ("WIN a FREE prize now!!! click http://bit.ly/xyz", 1),
    ("Congratulations! You won money, claim your FREE gift!!!", 1),
    ("URGENT: free offer, click here http://spam.link now!!!", 1),
    ("Cheap meds, buy now, limited free trial!!! http://deal.co", 1),
    ("Free money!!! click the link to win big http://win.biz", 1),
    ("Hi Sam, are we still on for lunch tomorrow?", 0),
    ("Please find the quarterly report attached. Thanks.", 0),
    ("Can you review my pull request when you get a chance?", 0),
    ("Reminder: team standup moved to 10am today.", 0),
    ("Thanks for dinner last night, it was great to catch up.", 0),
]


# ---------------------------------------------------------------------------
# GIVEN: tokenize -- chop text into lowercase word tokens (letters + digits).
# ---------------------------------------------------------------------------
def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


# ---------------------------------------------------------------------------
# TODO 1: build the VOCABULARY from the training emails.
# ---------------------------------------------------------------------------
def build_vocabulary(emails):
    """Return a dict {word: index} giving every distinct word its own slot.

    Example result: {"a": 0, "attached": 1, "big": 2, ...}
    The indices just need to be 0, 1, 2, ... with no gaps; order doesn't
    matter for the math. (The answer key sorts the words so the output is
    readable, but you don't have to.)

    Hint:
      - make an empty dict `vocab = {}`
      - loop over every (text, label) in `emails`
      - loop over every word in `tokenize(text)`
      - if the word is not already in `vocab`, add it with the NEXT index,
        which is just the current `len(vocab)`.
    """
    # >>> YOUR CODE HERE
    return {}  # replace this


# ---------------------------------------------------------------------------
# TODO 2: turn ONE email into a vector of word counts (bag-of-words).
# ---------------------------------------------------------------------------
def extract_features(text, vocab):
    """Return a list of length len(vocab): features[i] = count of word i.

    'free free win' -> the slots for 'free' and 'win' hold 2.0 and 1.0; every
    other slot stays 0.0. Words not in `vocab` are ignored.

    Hint:
      - start with `features = [0.0] * len(vocab)`
      - for each word in `tokenize(text)`, look up `idx = vocab.get(word)`
      - if `idx is not None`, do `features[idx] += 1.0`
      - return features
    """
    # >>> YOUR CODE HERE
    return [0.0] * len(vocab)  # replace this


# ---------------------------------------------------------------------------
# GIVEN: the neuron. Identical to Model 1 -- it doesn't care how many inputs.
# ---------------------------------------------------------------------------
def sigmoid(z):
    if z < -60:
        return 0.0
    if z > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


def predict_probability(features, weights, bias):
    z = bias
    for x, w in zip(features, weights):
        z += x * w
    return sigmoid(z)


# ---------------------------------------------------------------------------
# TODO 3: the learning loop. Same gradient descent as Model 1, but now there
# is one weight per vocabulary word instead of exactly three.
# ---------------------------------------------------------------------------
def train(emails, vocab, epochs=300, learning_rate=0.1, verbose=True):
    # TODO 3a: start every weight at 0.0 -- but how MANY weights now?
    #   Hint: one per word in the vocabulary -> [0.0] * len(vocab)
    weights = [0.0]  # >>> YOUR CODE HERE (fix the length)
    bias = 0.0

    dataset = [(extract_features(text, vocab), label) for text, label in emails]

    for epoch in range(epochs):
        total_loss = 0.0

        for features, truth in dataset:
            prediction = predict_probability(features, weights, bias)

            # TODO 3b: the learning rule (identical to Model 1).
            #   - error = prediction - truth
            #   - for each i: weights[i] -= learning_rate * error * features[i]
            #   - bias -= learning_rate * error
            # >>> YOUR CODE HERE
            error = 0.0  # replace with the real error, then do the updates

            p = min(max(prediction, 1e-12), 1 - 1e-12)
            total_loss += -(truth * math.log(p) + (1 - truth) * math.log(1 - p))

        if verbose and (epoch < 3 or (epoch + 1) % 100 == 0):
            print(f"epoch {epoch + 1:>4}: avg_loss={total_loss / len(dataset):.4f}")

    return weights, bias


# ---------------------------------------------------------------------------
# GIVEN: classify + a peek at what the model learned.
# ---------------------------------------------------------------------------
def classify(text, vocab, weights, bias):
    prob = predict_probability(extract_features(text, vocab), weights, bias)
    return prob, ("SPAM" if prob >= 0.5 else "not spam")


def show_learned_words(vocab, weights, top_n=6):
    index_to_word = {i: w for w, i in vocab.items()}
    ranked = sorted(range(len(weights)), key=lambda i: weights[i], reverse=True)
    print("\nTop SPAM words the model learned:")
    for i in ranked[:top_n]:
        print(f"  {weights[i]:+.3f}  {index_to_word[i]}")
    print("Top HAM words the model learned:")
    for i in reversed(ranked[-top_n:]):
        print(f"  {weights[i]:+.3f}  {index_to_word[i]}")


# ---------------------------------------------------------------------------
# SELF-CHECK: prove your model works. Don't edit this part.
# ---------------------------------------------------------------------------
def self_check():
    vocab = build_vocabulary(TRAINING_EMAILS)
    assert len(vocab) > 20, (
        f"TODO 1: vocabulary looks empty/too small ({len(vocab)} words). "
        "Did you fill in build_vocabulary?")
    assert "free" in vocab and "thanks" in vocab, (
        "TODO 1: expected words like 'free' and 'thanks' to be in the vocab.")

    feats = extract_features("free free win", vocab)
    assert len(feats) == len(vocab), "TODO 2: feature vector is the wrong length."
    assert feats[vocab["free"]] == 2.0, "TODO 2: 'free' should be counted twice."
    assert feats[vocab["win"]] == 1.0, "TODO 2: 'win' should be counted once."
    assert sum(feats) == 3.0, "TODO 2: only the words present should be nonzero."

    weights, bias = train(TRAINING_EMAILS, vocab, verbose=False)
    assert len(weights) == len(vocab), (
        "TODO 3a: you need one weight PER WORD -> [0.0] * len(vocab).")

    wrong = 0
    for text, truth in TRAINING_EMAILS:
        _, verdict = classify(text, vocab, weights, bias)
        if (verdict == "SPAM") != (truth == 1):
            wrong += 1
    assert wrong == 0, (
        f"TODO 3b: model got {wrong} training emails wrong -- check the update.")

    assert weights[vocab["free"]] > 0, "'free' should have learned a positive weight."
    print("\nALL CHECKS PASSED -- you rebuilt Model 2 from scratch.")


if __name__ == "__main__":
    # Run the self-check FIRST so an unfinished workbook prints a clear
    # "fill in TODO N" message instead of a confusing crash in the demo below.
    self_check()

    # Once the checks pass, enjoy the payoff: the model's learned words + demo.
    vocab = build_vocabulary(TRAINING_EMAILS)
    weights, bias = train(TRAINING_EMAILS, vocab)
    show_learned_words(vocab, weights)

    print("\nFresh emails:")
    for text in ["FREE money now!!! http://x.co", "can we move our 1:1 to Friday?"]:
        prob, verdict = classify(text, vocab, weights, bias)
        print(f"  p(spam)={prob:5.2f}  -> {verdict:8}  | {text}")
