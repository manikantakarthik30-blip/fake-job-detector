import numpy as np
import pandas as pd

from train import choose_threshold, load_data


def test_threshold_selection_returns_valid_probability():
    y = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.05, 0.10, 0.20, 0.65, 0.80, 0.90])
    threshold, rule = choose_threshold(y, probabilities)
    assert 0.0 <= threshold <= 1.0
    assert rule in {"max_f1", "max_recall_at_precision>=0.90", "default_no_thresholds"}


def test_load_data_drops_duplicates_and_handles_missing_text(tmp_path):
    frame = pd.DataFrame({
        "title": ["Engineer", "Engineer", "Analyst"],
        "description": ["Build systems", "Build systems", None],
        "fraudulent": [0, 0, 1],
    })
    path = tmp_path / "jobs.csv"
    frame.to_csv(path, index=False)
    X, y = load_data(path)
    assert len(X) == 2
    assert y.tolist() == [0, 1]
    assert X.iloc[1] == "Analyst"
