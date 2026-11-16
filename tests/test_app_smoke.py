import joblib
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest


def test_streamlit_single_listing_smoke(tmp_path, monkeypatch):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    samples = pd.Series(["software engineer salary benefits", "data analyst technical role",
                         "earn money registration fee", "guaranteed income telegram deposit"])
    labels = np.array([0, 0, 1, 1])
    pipe = Pipeline([("tfidf", TfidfVectorizer()), ("model", LogisticRegression(random_state=42))]).fit(samples, labels)
    artifact = tmp_path / "fixture.joblib"
    joblib.dump({"pipeline": pipe, "threshold": 0.5}, artifact)
    monkeypatch.setenv("FAKE_JOB_MODEL_PATH", str(artifact))
    app = AppTest.from_file("app.py", default_timeout=30).run()
    assert not app.exception
    app.text_input[0].set_value("Remote Data Entry")
    app.text_input[1].set_value("Example Hiring")
    app.text_area[0].set_value("Work from home. Pay a registration fee to begin.")
    app.button[0].click().run()
    assert not app.exception
    assert any("LIKELY" in item.value for item in app.markdown)
    assert any("Top 5 reasons" in item.value for item in app.markdown)
