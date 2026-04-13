"""Unit tests for finsense.evaluation.metrics."""

from finsense.evaluation.metrics import compute_metrics


def test_perfect_predictions():
    labels = [0, 1, 2, 0, 1, 2]
    preds = [0, 1, 2, 0, 1, 2]
    m = compute_metrics(labels, preds)
    assert m["macro_f1"] == 1.0
    assert m["per_class_f1"] == [1.0, 1.0, 1.0]
    assert m["balanced_accuracy"] == 1.0


def test_all_wrong():
    labels = [0, 0, 0]
    preds = [1, 1, 1]
    m = compute_metrics(labels, preds)
    assert m["macro_f1"] == 0.0


def test_partial_correct():
    labels = [0, 1, 2, 0, 1, 2]
    preds = [0, 1, 2, 1, 0, 2]  # 4/6 correct
    m = compute_metrics(labels, preds)
    assert 0.0 < m["macro_f1"] < 1.0
    assert len(m["per_class_f1"]) == 3
    # Neutral is perfect
    assert m["per_class_f1"][2] == 1.0


def test_confusion_matrix_shape():
    labels = [0, 1, 2]
    preds = [0, 1, 2]
    m = compute_metrics(labels, preds)
    cm = m["confusion_matrix"]
    assert len(cm) == 3
    assert all(len(row) == 3 for row in cm)


def test_confusion_matrix_diagonal():
    labels = [0, 1, 2, 0, 1, 2]
    preds = [0, 1, 2, 0, 1, 2]
    cm = compute_metrics(labels, preds)["confusion_matrix"]
    # All predictions on diagonal
    for i in range(3):
        for j in range(3):
            if i == j:
                assert cm[i][j] > 0
            else:
                assert cm[i][j] == 0
