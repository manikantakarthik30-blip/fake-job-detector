import joblib
import numpy as np
import pandas as pd

from predict import predict_one


def _save_fixture(path):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from imblearn.pipeline import Pipeline

    texts = pd.Series([
        "software engineer salary team benefits", "data analyst technical role",
        "python developer full time", "product manager company role",
        "earn money pay registration fee", "guaranteed income telegram deposit",
        "work from home unlimited income", "urgent wire transfer training fee",
        "legitimate engineering career", "pyramid scheme easy money",
    ])
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1])
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("model", LogisticRegression(random_state=42)),
    ]).fit(texts, labels)
    joblib.dump({"pipeline": pipe, "threshold": .5, "model_name": "test"}, path)


def test_model_loads_probability_range_and_contract(tmp_path):
    path = tmp_path / "model.joblib"
    _save_fixture(path)
    result = predict_one({"title": "Urgent job", "description": "Pay deposit fee"}, str(path))
    assert 0.0 <= result["probability"] <= 1.0
    assert 0.0 < result["threshold"] < 1.0
    assert result["verdict"] in {"LIKELY_FRAUDULENT", "LIKELY_LEGITIMATE"}
    assert isinstance(result["top_signals"], list)


def test_prediction_is_deterministic(tmp_path):
    path = tmp_path / "model.joblib"
    _save_fixture(path)
    listing = {"title": "Data entry", "description": "Work from home, pay registration fee"}
    assert predict_one(listing, str(path)) == predict_one(listing, str(path))
