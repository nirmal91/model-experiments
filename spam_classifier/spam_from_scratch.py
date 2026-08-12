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
    "dollars"
}

class Model:
    def __init__(self):
        self.weights = [0.0, 0.0, 0.0]
        self.bias = 0.0

def get_features(text):
    lower = text.lower()
    words = lower.replace("!", " ").replace(",", " ").split()
    return [
        sum(1 for w in words if w in SPAM_WORDS),
        text.count("!"),
        1.0 if ("http" in lower or "www" in lower) else 0.0
    ]

def sigmoid(prob):
    return 1.0 / (1.0 + math.exp(-prob))

def predict_probability(features, model):
    feature_weight_sum = sum(feature * weight for feature, weight in zip(features, model.weights)) + model.bias
    return sigmoid(feature_weight_sum)

def build_model():
    """
    basically, a model object is a bunch of weights and a bias here. We can choose 3 weights here. 

    """
    model = Model()
    epochs = 200
    learning_rate = 0.1
    for epoch in range(epochs):
        loss = 0.0
        for email, label in TRAINING_EMAILS:
            features = get_features(email)
            predicted_prob = predict_probability(features, model)
            error = label - predicted_prob

            # based on error, adjust weights and biases
            """
            if the error is positive, our features were too low, so we need to increase weights and biases
            if the error is negative, our features were too high, so we need to decrease weights and biases
            """
            for idx, feature in enumerate(features):
                model.weights[idx] += learning_rate * error * feature
            model.bias += learning_rate * error

            # simple loss: how far off were we? (absolute error)
            loss += abs(error)

        # print the loss and the weights and biases
        avg_loss = loss / len(TRAINING_EMAILS)
        if epoch < 5 or (epoch + 1) % 40 == 0:
            print(f"epoch {epoch + 1}: avg_loss={avg_loss:.4f}  weights={model.weights}  bias={model.bias:.4f}")

    print(model.weights)
    print(model.bias)

    return model

def is_spam_or_not(model, email):
    features = get_features(email)
    prob = predict_probability(features, model)
    if prob > 0.5:
        return True
    else:
        return False


if __name__ == "__main__":
    model = build_model()
    print(is_spam_or_not(model, "Free Pizza!!!"))
    print(is_spam_or_not(model, "Hello from Mercury"))
    print(is_spam_or_not(model, "DOLLARS!!"))
