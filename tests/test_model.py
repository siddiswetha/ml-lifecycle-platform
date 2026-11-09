import math

from ml_platform import data as data_module
from ml_platform.features import engineer_all
from ml_platform.model import (
    evaluate_metrics,
    predict,
    predict_proba,
    sigmoid,
    train_logistic_regression,
)


def _trained():
    rows = data_module.generate_synthetic(n=200, seed=7)
    train, held = data_module.split(rows, test_size=0.2, seed=7)
    train = engineer_all(train)
    held = engineer_all(held)
    model = train_logistic_regression(train, [r["will_fail_30d"] for r in train], epochs=100)
    return model, held


def test_sigmoid_midpoint():
    assert sigmoid(0) == 0.5


def test_sigmoid_extremes():
    assert sigmoid(100) > 0.999
    assert sigmoid(-100) < 0.001


def test_train_returns_nine_weights():
    model, _ = _trained()
    assert len(model["weights"]) == 9
    assert all(math.isfinite(w) for w in model["weights"])
    assert math.isfinite(model["bias"])


def test_predict_proba_in_range():
    model, held = _trained()
    probs = predict_proba(model, held)
    assert len(probs) == len(held)
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_predict_labels_binary():
    model, held = _trained()
    assert set(predict(model, held)) <= {0, 1}


def test_evaluate_metrics_perfect():
    metrics = evaluate_metrics([1, 1, 0, 0], [1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1])
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["auc"] == 1.0


def test_evaluate_metrics_counts():
    metrics = evaluate_metrics([1, 0, 1, 0], [1, 1, 0, 0])
    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tn"] == 1


def test_train_rejects_empty():
    try:
        train_logistic_regression([], [])
    except ValueError:
        return
    raise AssertionError("expected ValueError")
