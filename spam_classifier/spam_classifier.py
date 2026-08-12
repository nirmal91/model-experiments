"""
The simplest possible "machine that learns": a spam / not-spam classifier.
=========================================================================

This is a single-layer linear model trained with logistic regression.
It is about as basic as a learning algorithm gets. It is NOT a deep neural
network -- it's ONE neuron: it multiplies a few inputs by a few weights,
adds them up, and squashes the result into a probability. Training is just a
`for` loop that nudges the weights a little each time it sees an example.

No libraries. Only the Python standard library. Every number is visible.

The 5 things we do (matching the plan):
  1. Go through a lot of examples (a small labeled dataset of emails).
  2. Learn just THREE weights (+ one bias), one per feature.
  3. Compute the weighted sum and squash it into a probability (the "score").
  4. Feed in a FRESH email and see if the model calls it spam.
  5. A plain `for` loop of learning -- and we watch the weights converge.
"""

import math


# ---------------------------------------------------------------------------
# STEP 1: The examples.
# ---------------------------------------------------------------------------
# Each example is (email_text, label). label = 1 means SPAM, 0 means NOT spam.
# These are deliberately obvious so you can see the model "get it".
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
# STEP 2 (part A): Turn an email into THREE numbers -- the "features".
# ---------------------------------------------------------------------------
# The model can't read English. It only sees numbers. So we hand it three
# simple, intuitive signals. Each will get exactly ONE weight.
SPAM_WORDS = {
    "free", "win", "won", "prize", "money", "offer", "click",
    "buy", "cheap", "urgent", "claim", "gift", "trial", "limited",
}


def extract_features(text):
    """Return [x1, x2, x3] for one email.

    x1 = how many "spammy" words appear   (free, win, click, ...)
    x2 = how many exclamation marks '!'    (spam LOVES these!!!)
    x3 = does it contain a link?  1 or 0   (http / www)
    """
    lower = text.lower()
    words = lower.replace("!", " ").replace(",", " ").split()

    x1 = sum(1 for w in words if w in SPAM_WORDS)
    x2 = text.count("!")
    x3 = 1.0 if ("http" in lower or "www" in lower) else 0.0

    return [float(x1), float(x2), float(x3)]


# ---------------------------------------------------------------------------
# STEP 3: The model itself -- weighted sum, then squashed to a probability.
# ---------------------------------------------------------------------------
def sigmoid(z):
    """Squash any number into the range (0, 1) so we can read it as a
    probability. Big positive z -> near 1 (spam). Big negative -> near 0."""
    # Guard against math.exp overflow for very negative z.
    if z < -60:
        return 0.0
    if z > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


def predict_probability(features, weights, bias):
    """The core computation: multiply each feature by its weight, add them
    up, add the bias, then squash. This single line IS the model."""
    z = bias
    for x, w in zip(features, weights):
        z += x * w
    return sigmoid(z)


# ---------------------------------------------------------------------------
# STEP 5: The learning loop -- a plain `for` loop that makes weights converge.
# ---------------------------------------------------------------------------
def train(emails, epochs=200, learning_rate=0.1, verbose=True):
    """Gradient descent, by hand.

    Start with all weights = 0 (the model knows nothing). Then, over and over:
      - look at an example,
      - see how wrong the prediction is (error = prediction - truth),
      - nudge each weight a little in the direction that reduces the error.
    Do this enough times and the weights settle down -- they CONVERGE.
    """
    # Three features -> three weights. Plus one bias (the "default lean").
    weights = [0.0, 0.0, 0.0]
    bias = 0.0

    # Pre-compute the feature vectors once.
    dataset = [(extract_features(text), label) for text, label in emails]

    if verbose:
        print("Watching the weights converge (one line per checkpoint):\n")
        print(f"{'epoch':>6} | {'w1(spam words)':>14} | {'w2(!marks)':>11} | "
              f"{'w3(link)':>9} | {'bias':>7} | {'avg loss':>8}")
        print("-" * 74)

    for epoch in range(epochs):
        total_loss = 0.0

        # --- one full pass over every example ---
        for features, truth in dataset:
            prediction = predict_probability(features, weights, bias)

            # How wrong were we? (positive => we over-predicted spam)
            error = prediction - truth

            # Nudge every weight down the gradient. This is the whole "learning".
            for i in range(len(weights)):
                weights[i] -= learning_rate * error * features[i]
            bias -= learning_rate * error

            # Track loss just so we can SEE it shrinking (log loss).
            p = min(max(prediction, 1e-12), 1 - 1e-12)
            total_loss += -(truth * math.log(p) + (1 - truth) * math.log(1 - p))

        # Print a checkpoint now and then so the convergence is visible.
        if verbose and (epoch < 5 or (epoch + 1) % 40 == 0):
            avg_loss = total_loss / len(dataset)
            print(f"{epoch + 1:>6} | {weights[0]:>14.4f} | {weights[1]:>11.4f} | "
                  f"{weights[2]:>9.4f} | {bias:>7.4f} | {avg_loss:>8.4f}")

    return weights, bias


# ---------------------------------------------------------------------------
# STEP 4: Use the trained model on FRESH emails it has never seen.
# ---------------------------------------------------------------------------
def classify(text, weights, bias):
    features = extract_features(text)
    prob = predict_probability(features, weights, bias)
    verdict = "SPAM" if prob >= 0.5 else "not spam"
    return prob, verdict, features


def main():
    print("=" * 74)
    print("A very basic spam classifier (one neuron, trained by a for loop)")
    print("=" * 74, "\n")

    # ---- Learn the weights from the examples ----
    weights, bias = train(TRAINING_EMAILS, epochs=200, learning_rate=0.1)

    print("\nFinal learned weights:")
    print(f"  w1 (spam-word count) = {weights[0]:+.4f}  "
          "<- more spammy words pushes toward SPAM")
    print(f"  w2 (! count)         = {weights[1]:+.4f}  "
          "<- more '!' pushes toward SPAM")
    print(f"  w3 (has a link)      = {weights[2]:+.4f}  "
          "<- having a link pushes toward SPAM")
    print(f"  bias                 = {bias:+.4f}  "
          "<- the model's default lean when all features are 0")

    # ---- Sanity check: did it learn the training set? ----
    print("\nHow it scores the emails it trained on:")
    print("-" * 74)
    for text, truth in TRAINING_EMAILS:
        prob, verdict, _ = classify(text, weights, bias)
        truth_label = "SPAM" if truth == 1 else "not spam"
        mark = "OK " if verdict == truth_label else "XX "
        print(f"  {mark} p(spam)={prob:5.2f}  guess={verdict:8}  "
              f"truth={truth_label:8}  | {text[:34]}")

    # ---- STEP 4: brand-new emails the model has NEVER seen ----
    print("\nFresh, unseen emails:")
    print("-" * 74)
    fresh_emails = [
        "FREE prize!!! click now http://totally.legit",   # should be SPAM
        "Hey, can we reschedule our 1:1 to Thursday?",     # should be not spam
        "Urgent: claim your free money now!!!",            # should be SPAM
        "The build passed, merging the PR now.",           # should be not spam
    ]
    for text in fresh_emails:
        prob, verdict, feats = classify(text, weights, bias)
        print(f"  p(spam)={prob:5.2f}  -> {verdict:8}  "
              f"features(words,!,link)={feats}  | {text}")

    print("\nDone. Try editing TRAINING_EMAILS or the fresh_emails and re-run.")


if __name__ == "__main__":
    main()
