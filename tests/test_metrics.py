import json

from src.evaluation.metrics import evaluate_predictions


def test_evaluate_predictions_perfect_classifier():
    y_true = ["R", "DoS", "R", "DoS"]
    y_pred = ["R", "DoS", "R", "DoS"]

    report = evaluate_predictions(y_true, y_pred, labels=("DoS", "R"))

    assert report.accuracy == 1.0
    assert report.precision == 1.0
    assert report.recall == 1.0
    assert report.f1_score == 1.0
    assert report.confusion_matrix.tolist() == [[2, 0], [0, 2]]


def test_evaluate_predictions_all_wrong():
    y_true = ["R", "R"]
    y_pred = ["DoS", "DoS"]

    report = evaluate_predictions(y_true, y_pred, labels=("DoS", "R"))

    assert report.accuracy == 0.0
    assert report.recall == 0.0


def test_evaluate_predictions_to_dict_is_json_serializable():
    report = evaluate_predictions(["R"], ["R"], labels=("R",))
    payload = json.dumps(report.to_dict())
    assert "accuracy" in payload
