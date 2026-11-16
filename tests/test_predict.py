import joblib
import numpy as np
import pandas as pd
import pytest
from predict import predict_one, predict_many


def make_artifact(path):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    texts = pd.Series(["software engineer salary", "data analyst technical", "python developer role",
                       "earn money registration fee", "guaranteed income telegram", "work from home unlimited income"])
    labels = np.array([0, 0, 0, 1, 1, 1])
    pipe = Pipeline([("tfidf", TfidfVectorizer()), ("model", LogisticRegression(random_state=42))]).fit(texts, labels)
    joblib.dump({"pipeline": pipe, "threshold": 0.5}, path)


def test_prediction_contract_and_batch(tmp_path):
    path = tmp_path / "fixture.joblib"
    make_artifact(path)
    row = {"title": "Data entry", "description": "Work from home, pay registration fee"}
    result = predict_one(row, str(path))
    assert 0 <= result["probability"] <= 1
    assert result["verdict"] in {"LIKELY_FRAUDULENT", "LIKELY_LEGITIMATE"}
    assert len(predict_many([row], str(path))) == 1
    assert result == predict_one(row, str(path))


def test_empty_input_rejected():
    with pytest.raises(ValueError, match="Enter a job title"):
        predict_one({})
