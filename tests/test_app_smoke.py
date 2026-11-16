from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest


def _write_tiny_model(path):
    from imblearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    samples = pd.Series([
        "software engineer salary company benefits",
        "data analyst full time technical team",
        "python developer professional experience",
        "earn money pay registration fee",
        "guaranteed income telegram deposit required",
        "work from home unlimited income",
    ])
    labels = np.array([0, 0, 0, 1, 1, 1])
    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("model", LogisticRegression(random_state=42)),
    ]).fit(samples, labels)
    joblib.dump({"pipeline": model, "threshold": 0.5, "model_name": "test"}, path)


def test_streamlit_single_check_end_to_end(tmp_path):
    root = Path(__file__).resolve().parents[1]
    artifact = root / "model.joblib"
    backup = tmp_path / "previous_model.joblib"
    if artifact.exists():
        backup.write_bytes(artifact.read_bytes())
    try:
        _write_tiny_model(artifact)
        app = AppTest.from_file(str(root / "app.py"), default_timeout=30).run()
        assert not app.exception

        app.text_input[0].set_value("Remote Data Entry")
        app.text_input[1].set_value("Example Hiring")
        app.text_area[0].set_value(
            "Work from home. No experience needed. Pay a registration fee to begin."
        )
        app.button[0].click().run()
        assert not app.exception
        rendered = " ".join(element.value for element in app.markdown)
        assert "LIKELY" in rendered
        assert "%" in rendered
    finally:
        if backup.exists():
            artifact.write_bytes(backup.read_bytes())
        elif artifact.exists():
            artifact.unlink()
