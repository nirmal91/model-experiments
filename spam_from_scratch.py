"""
WORKBOOK: build the spam classifier yourself.
==============================================

Fill in every spot marked  # >>> YOUR CODE HERE.
Each blank has a hint telling you exactly what one or two lines to write.

Run it any time:   python3 spam_from_scratch.py
While blanks are empty it will fail or misbehave -- that's expected. Keep
filling them in and re-running. When the self-check at the bottom prints
"ALL CHECKS PASSED", you've rebuilt the whole model from scratch.

If you get stuck, peek at the finished version in spam_classifier.py.
"""

import math


# ---------------------------------------------------------------------------
# GIVEN: the examples and the feature extractor. Nothing to do here -- read it.
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

SPAM_WORDS = {
    "free", "win", "won", "prize", "money", "offer", "click",
    "buy", "cheap", "urgent", "claim", "gift", "trial", "limited",
}


def extract_features(text):
    """Turn one email into [x1, x2, x3]. (Given -- just read it.)"""
    lower = text.lower()
    words = lower.replace("!", " ").replace(",", " ").split()
    x1 = sum(1 for w in words if w in SPAM_WORDS)   # spam-word count
    x2 = text.count("!")                            # exclamation marks
    x3 = 1.0 if ("http" in lower or "www" in lower) else 0.0  # has a link?
    return [float(x1), float(x2), float(x3)]


# ===========================================================================
# TASK 1: the sigmoid -- squash any number into (0, 1).
# ===========================================================================
def sigmoid(z):
    # Formula:  1 / (1 + e^(-z))
    # In Python, e^(-z) is  math.exp(-z).
    # (Optional: if z < -60 return 0.0 and if z > 60 return 1.0 to avoid
    #  math.exp overflowing. Not required for the model to work.)
    #
    # >>> YOUR CODE HERE: return the sigmoid of z
    pass


# ===========================================================================
# TASK 2: the model -- weighted sum, then squash.
# ===========================================================================
def predict_probability(features, weights, bias):
    # 1. Start a running total `z` at the bias.
    # 2. For each (feature x, weight w) pair, add x * w to z.
    #    Hint: for x, w in zip(features, weights):
    # 3. Return sigmoid(z).
    #
    # >>> YOUR CODE HERE (about 4 lines)
    pass


# ===========================================================================
# TASK 3: the learning loop.
# ===========================================================================
def train(emails, epochs=200, learning_rate=0.1, verbose=True):
    weights = [0.0, 0.0, 0.0]   # start knowing nothing
    bias = 0.0

    dataset = [(extract_features(text), label) for text, label in emails]

    if verbose:
        print(f"{'epoch':>6} | {'w1':>7} | {'w2':>7} | {'w3':>7} | "
              f"{'bias':>7} | {'avg loss':>8}")
        print("-" * 56)

    for epoch in range(epochs):
        total_loss = 0.0   # reset the scoreboard each epoch

        for features, truth in dataset:
            # (a) Ask the model what it currently thinks.
            #     prediction = predict_probability(features, weights, bias)
            #
            # >>> YOUR CODE HERE: compute `prediction`
            prediction = None  # replace this

            # (b) How wrong were we?  error = prediction - truth
            #
            # >>> YOUR CODE HERE: compute `error`
            error = None  # replace this

            # (c) Nudge each weight:  weights[i] -= learning_rate * error * features[i]
            #     Then nudge bias:    bias       -= learning_rate * error
            #
            # >>> YOUR CODE HERE: update every weight in a loop, then the bias

            # (d) Scoreboard only (given). Adds this email's "badness".
            p = min(max(prediction, 1e-12), 1 - 1e-12)
            total_loss += -(truth * math.log(p) + (1 - truth) * math.log(1 - p))

        if verbose and (epoch < 5 or (epoch + 1) % 40 == 0):
            avg_loss = total_loss / len(dataset)
            print(f"{epoch + 1:>6} | {weights[0]:>7.3f} | {weights[1]:>7.3f} | "
                  f"{weights[2]:>7.3f} | {bias:>7.3f} | {avg_loss:>8.4f}")

    return weights, bias


# ===========================================================================
# GIVEN: use the trained model. Nothing to fill in.
# ===========================================================================
def classify(text, weights, bias):
    prob = predict_probability(extract_features(text), weights, bias)
    return prob, ("SPAM" if prob >= 0.5 else "not spam")


# ===========================================================================
# SELF-CHECK: run everything and verify your code behaves correctly.
# ===========================================================================
def self_check():
    problems = []

    # sigmoid sanity
    if sigmoid(0) is None or abs(sigmoid(0) - 0.5) > 1e-9:
        problems.append("sigmoid(0) should be 0.5 -- check TASK 1")

    try:
        weights, bias = train(TRAINING_EMAILS, verbose=True)
    except TypeError:
        print("\n" + "=" * 56)
        print("Got a TypeError -- that means a blank is still returning None.")
        print("Fill in TASK 1, 2, and 3, then run again.")
        print("=" * 56)
        return

    # After training, every training email should be classified correctly.
    wrong = 0
    for text, truth in TRAINING_EMAILS:
        _, verdict = classify(text, weights, bias)
        if (verdict == "SPAM") != (truth == 1):
            wrong += 1
    if wrong:
        problems.append(f"{wrong} training emails misclassified -- check TASK 2 & 3")

    # A fresh spammy email should score high.
    p_spam, _ = classify("FREE money!!! click http://x.co", weights, bias)
    if p_spam < 0.8:
        problems.append(f"fresh spam scored only {p_spam:.2f} (want > 0.8)")

    print("\n" + "=" * 56)
    if problems:
        print("Not passing yet:")
        for pr in problems:
            print("  -", pr)
    else:
        print("ALL CHECKS PASSED -- you rebuilt the model from scratch. Nice.")
    print("=" * 56)


if __name__ == "__main__":
    self_check()
