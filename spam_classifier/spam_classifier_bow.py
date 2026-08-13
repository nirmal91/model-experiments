"""
The next model: bag-of-words logistic regression.
==================================================

Same machine as spam_classifier.py -- ONE neuron, a sigmoid, and a plain `for`
loop of gradient descent. We change exactly ONE thing: the input.

Before (spam_classifier.py):
    You hand-picked 3 features and hand-wrote a SPAM_WORDS list. The model
    only learned HOW MUCH each of your 3 features mattered.

Now (this file):
    We build a VOCABULARY automatically from the training emails, and turn each
    email into a vector of word counts -- one number per word. The model gets
    ONE weight per word and LEARNS which words are spammy. No SPAM_WORDS list.

That's the whole idea: don't tell the model that "free" is spammy. Give it a
weight for every word and let training discover it. After training we can sort
the weights and literally read off the words the model decided are spam signals.

Still no libraries. Only the Python standard library.
"""

import math
import re


# ---------------------------------------------------------------------------
# STEP 1: The examples. (Unchanged from the first model.)
# ---------------------------------------------------------------------------
# label = 1 means SPAM, 0 means NOT spam (ham).
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
# STEP 2 (part A): Tokenize -- chop text into words the model can count.
# ---------------------------------------------------------------------------
def tokenize(text):
    """Lowercase, then pull out 'word' tokens.

    A URL like http://bit.ly/xyz becomes the token 'http' plus some junk, which
    is fine -- 'http' itself turns out to be a useful spam signal. We keep this
    deliberately dumb: letters and digits only, everything else is a separator.
    """
    return re.findall(r"[a-z0-9]+", text.lower())


# ---------------------------------------------------------------------------
# STEP 2 (part B): Build the VOCABULARY from the training emails.
# ---------------------------------------------------------------------------
# This is the key new step. Instead of a hand-written SPAM_WORDS set, we scan
# every training email and collect the words we see. Each word gets a fixed
# position (an index) in the feature vector. Word #0 always maps to slot 0, etc.
def build_vocabulary(emails, min_count=1):
    """Return {word: index}. Words seen >= min_count times get a slot.

    min_count is a knob: raise it to drop rare words (fewer weights, less
    memorization). With this tiny dataset we keep min_count=1 so nothing is lost.
    """
    counts = {}
    for text, _label in emails:
        for word in tokenize(text):
            counts[word] = counts.get(word, 0) + 1

    # Sort for a stable, readable ordering (not required by the math).
    vocab = {}
    for word in sorted(counts):
        if counts[word] >= min_count:
            vocab[word] = len(vocab)
    return vocab


# ---------------------------------------------------------------------------
# STEP 2 (part C): Turn ONE email into a vector of word counts (bag-of-words).
# ---------------------------------------------------------------------------
def extract_features(text, vocab):
    """Return a list of length len(vocab): features[i] = count of word i.

    'free free win' -> the slots for 'free' and 'win' get 2 and 1; every other
    slot stays 0. Order is thrown away -- it's a *bag* of words, not a sequence.
    Words not in the vocabulary (unseen at training time) are simply ignored.
    """
    features = [0.0] * len(vocab)
    for word in tokenize(text):
        idx = vocab.get(word)
        if idx is not None:
            features[idx] += 1.0
    return features


# ---------------------------------------------------------------------------
# STEP 3: The model -- weighted sum, then squashed to a probability.
# ---------------------------------------------------------------------------
# IDENTICAL to the first model. The neuron does not know or care that it now has
# hundreds of inputs instead of three.
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
# STEP 5: The learning loop. ALSO identical -- one weight per feature, and the
# feature count just happens to be len(vocab) now instead of 3.
# ---------------------------------------------------------------------------
def train(emails, vocab, epochs=300, learning_rate=0.1, verbose=True):
    """Gradient descent, by hand -- the exact same rule as before.

      prediction = sigmoid(w . x + bias)
      error      = prediction - truth
      w[i]      -= rate * error * x[i]     # one nudge per feature
      bias      -= rate * error

    The only difference from the 3-feature model: `weights` starts as a list of
    len(vocab) zeros instead of [0, 0, 0].
    """
    weights = [0.0] * len(vocab)
    bias = 0.0

    dataset = [(extract_features(text, vocab), label) for text, label in emails]

    if verbose:
        print(f"Vocabulary size: {len(vocab)} words  ->  {len(vocab)} weights "
              f"(the old model had 3)\n")
        print(f"{'epoch':>6} | {'avg loss':>9} | {'||weights||':>11}")
        print("-" * 34)

    for epoch in range(epochs):
        total_loss = 0.0

        for features, truth in dataset:
            prediction = predict_probability(features, weights, bias)
            error = prediction - truth

            # Only the slots that actually appear in this email are non-zero,
            # so only those weights move. (error * 0 == 0 for absent words.)
            for i in range(len(weights)):
                weights[i] -= learning_rate * error * features[i]
            bias -= learning_rate * error

            p = min(max(prediction, 1e-12), 1 - 1e-12)
            total_loss += -(truth * math.log(p) + (1 - truth) * math.log(1 - p))

        if verbose and (epoch < 5 or (epoch + 1) % 50 == 0):
            avg_loss = total_loss / len(dataset)
            norm = math.sqrt(sum(w * w for w in weights))
            print(f"{epoch + 1:>6} | {avg_loss:>9.4f} | {norm:>11.4f}")

    return weights, bias


# ---------------------------------------------------------------------------
# STEP 4: Use the trained model on fresh emails.
# ---------------------------------------------------------------------------
def classify(text, vocab, weights, bias):
    features = extract_features(text, vocab)
    prob = predict_probability(features, weights, bias)
    verdict = "SPAM" if prob >= 0.5 else "not spam"
    return prob, verdict


# ---------------------------------------------------------------------------
# The payoff: read the model's mind. Which words did it decide are spammy?
# ---------------------------------------------------------------------------
def show_learned_words(vocab, weights, top_n=8):
    """Sort words by their learned weight. Big positive = spam signal, big
    negative = ham signal. THIS is what the model learned instead of us
    hand-writing SPAM_WORDS."""
    index_to_word = {i: w for w, i in vocab.items()}
    ranked = sorted(range(len(weights)), key=lambda i: weights[i], reverse=True)

    print("\nWords the model learned to associate with SPAM (highest weights):")
    for i in ranked[:top_n]:
        print(f"  {weights[i]:+.3f}  {index_to_word[i]}")

    print("\nWords the model learned to associate with HAM (lowest weights):")
    for i in reversed(ranked[-top_n:]):
        print(f"  {weights[i]:+.3f}  {index_to_word[i]}")


def main():
    print("=" * 70)
    print("Bag-of-words spam classifier (one neuron, one weight per word)")
    print("=" * 70, "\n")

    # ---- Build the vocabulary FROM the data (no hand-written word list) ----
    vocab = build_vocabulary(TRAINING_EMAILS)

    # ---- Learn one weight per word ----
    weights, bias = train(TRAINING_EMAILS, vocab, epochs=300, learning_rate=0.1)

    # ---- The whole point: the model discovered its own spam words ----
    show_learned_words(vocab, weights)

    # ---- Sanity check on the training set ----
    print("\nHow it scores the emails it trained on:")
    print("-" * 70)
    for text, truth in TRAINING_EMAILS:
        prob, verdict = classify(text, vocab, weights, bias)
        truth_label = "SPAM" if truth == 1 else "not spam"
        mark = "OK " if verdict == truth_label else "XX "
        print(f"  {mark} p(spam)={prob:5.2f}  guess={verdict:8}  | {text[:40]}")

    # ---- Fresh, unseen emails ----
    print("\nFresh, unseen emails:")
    print("-" * 70)
    fresh_emails = [
        "FREE prize!!! click now http://totally.legit",   # should be SPAM
        "Hey, can we reschedule our 1:1 to Thursday?",     # should be not spam
        "Urgent: claim your free money now!!!",            # should be SPAM
        "The build passed, merging the PR now.",           # should be not spam
    ]
    for text in fresh_emails:
        prob, verdict = classify(text, vocab, weights, bias)
        print(f"  p(spam)={prob:5.2f}  -> {verdict:8}  | {text}")

    # ---- The honest catch: memorization vs. generalization ----
    print("\n" + "-" * 70)
    print("Note: with a big vocabulary and only 10 emails, near-zero loss means")
    print("the model can MEMORIZE the training set. That no longer proves it")
    print("generalizes. The next step is a train/test split to measure that.")


if __name__ == "__main__":
    main()
