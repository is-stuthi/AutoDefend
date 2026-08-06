import numpy as np
import pytest

from src.models.random_forest import RandomForestDetector


def _toy_dataset() -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(42)
    normal = rng.normal(loc=0.0, scale=1.0, size=(50, 3))
    attack = rng.normal(loc=10.0, scale=1.0, size=(50, 3))
    X = np.vstack([normal, attack])
    y = ["R"] * 50 + ["DoS"] * 50
    return X, y


def test_train_predict_roundtrip_separates_classes():
    X, y = _toy_dataset()
    model = RandomForestDetector(n_estimators=20, random_state=0)
    model.train(X, y, feature_names=["a", "b", "c"])

    predictions = model.predict(X)
    accuracy = float(np.mean(np.array(predictions) == np.array(y)))
    assert accuracy > 0.95


def test_predict_before_training_raises():
    model = RandomForestDetector()
    with pytest.raises(RuntimeError):
        model.predict(np.zeros((1, 3)))


def test_save_and_load_model_roundtrip(tmp_path):
    X, y = _toy_dataset()
    model = RandomForestDetector(n_estimators=20, random_state=0)
    model.train(X, y, feature_names=["a", "b", "c"])
    original_predictions = model.predict(X)

    model_path = tmp_path / "model.joblib"
    model.save_model(model_path)

    loaded_model = RandomForestDetector()
    loaded_model.load_model(model_path)
    loaded_predictions = loaded_model.predict(X)

    assert list(original_predictions) == list(loaded_predictions)
    assert set(loaded_model.feature_importances()) == {"a", "b", "c"}


def test_load_missing_model_file_raises(tmp_path):
    model = RandomForestDetector()
    with pytest.raises(FileNotFoundError):
        model.load_model(tmp_path / "does_not_exist.joblib")
