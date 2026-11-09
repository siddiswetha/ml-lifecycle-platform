"""Logistic regression implemented from scratch (standard library only).

The model is a small, auditable binary classifier: features are
standardised with the training mean / standard deviation, then a linear
score is passed through a sigmoid. Training is batch gradient descent
with L2 regularisation. Weights are stored as plain JSON so the
registry can version them without any binary serialisation format.
"""

import json
import math

from .features import ALL_FEATURES

# Five raw sensor features plus four engineered features (see features.py).
FEATURE_ORDER = list(ALL_FEATURES)


def sigmoid(z):
    """Numerically stable sigmoid."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _std(values, mean):
    if len(values) < 2:
        return 1.0
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(var) if var > 0 else 1.0


def fit_scaler(rows):
    """Return {feature: {"mean": ..., "std": ...}} from training rows."""
    scaler = {}
    for feature in FEATURE_ORDER:
        values = [float(r[feature]) for r in rows]
        mean = _mean(values)
        scaler[feature] = {"mean": mean, "std": _std(values, mean)}
    return scaler


def _standardise(row, scaler):
    return [
        (float(row[f]) - scaler[f]["mean"]) / scaler[f]["std"]
        for f in FEATURE_ORDER
    ]


def train_logistic_regression(rows, labels, epochs=300, learning_rate=0.05, l2=0.01):
    """Train by batch gradient descent. Returns a JSON-serialisable model dict."""
    if not rows:
        raise ValueError("no training rows")
    if len(rows) != len(labels):
        raise ValueError("rows and labels have different lengths")
    scaler = fit_scaler(rows)
    x = [_standardise(r, scaler) for r in rows]
    y = [float(v) for v in labels]
    n = len(rows)
    weights = [0.0] * len(FEATURE_ORDER)
    bias = 0.0
    for _ in range(epochs):
        grad_w = [0.0] * len(FEATURE_ORDER)
        grad_b = 0.0
        for xi, yi in zip(x, y):
            z = sum(w * v for w, v in zip(weights, xi)) + bias
            error = sigmoid(z) - yi
            for j in range(len(weights)):
                grad_w[j] += error * xi[j]
            grad_b += error
        for j in range(len(weights)):
            weights[j] -= learning_rate * (grad_w[j] / n + l2 * weights[j])
        bias -= learning_rate * (grad_b / n)
    return {
        "type": "logistic_regression",
        "features": list(FEATURE_ORDER),
        "weights": weights,
        "bias": bias,
        "scaler": scaler,
        "epochs": epochs,
        "learning_rate": learning_rate,
    }


def predict_proba(model, rows):
    """Return a probability per row."""
    scaler = model["scaler"]
    weights = model["weights"]
    bias = model["bias"]
    out = []
    for row in rows:
        xi = _standardise(row, scaler)
        z = sum(w * v for w, v in zip(weights, xi)) + bias
        out.append(sigmoid(z))
    return out


def predict(model, rows, threshold=0.5):
    return [1 if p >= threshold else 0 for p in predict_proba(model, rows)]


def _auc(y_true, y_score):
    """AUC by the rank (Mann-Whitney) method; returns None if undefined."""
    positives = sum(1 for v in y_true if v == 1)
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return None
    order = sorted(range(len(y_true)), key=lambda i: y_score[i])
    ranks = [0.0] * len(y_true)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and y_score[order[j + 1]] == y_score[order[i]]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    rank_sum = sum(ranks[i] for i, v in enumerate(y_true) if v == 1)
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def evaluate_metrics(y_true, y_pred, y_score=None):
    """Accuracy, precision, recall, F1 and (when scores are given) AUC."""
    if not y_true:
        raise ValueError("no labels to evaluate")
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    n = len(y_true)
    accuracy = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    result = {
        "n": n,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
    if y_score is not None:
        auc = _auc(y_true, y_score)
        result["auc"] = round(auc, 4) if auc is not None else None
    return result


def save_model(model, path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(model, fh, indent=2)


def load_model(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
