"""Single-listing inference using the saved training pipeline."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re

import joblib
import pandas as pd


@lru_cache(maxsize=2)
def _load_model(path):
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {artifact_path}")
    return joblib.load(artifact_path)


def _combine(text_dict):
    fields = ("title", "company_profile", "description", "requirements", "benefits",
              "location", "employment_type", "required_experience", "required_education")
    raw = " ".join(str(text_dict.get(field) or "") for field in fields).lower()
    raw = re.sub(r"<[^>]+>", " ", raw)
    raw = re.sub(r"https?://\S+|www\.\S+", " ", raw)
    return re.sub(r"\s+", " ", raw).strip()


def predict_one(text_dict, model_path="model.joblib"):
    """Return fraud probability, thresholded verdict, and top TF-IDF signals."""
    artifact = _load_model(str(Path(model_path).resolve()))
    text = _combine(text_dict)
    probability = float(artifact["pipeline"].predict_proba(pd.Series([text]))[0, 1])
    threshold = float(artifact["threshold"])
    pipe = artifact["pipeline"]
    vectorizer = pipe.named_steps["tfidf"]
    classifier = pipe.named_steps["model"]
    top_signals = []
    # Coefficients have direct term meaning for Logistic Regression only.
    if hasattr(classifier, "coef_"):
        tokens = set(vectorizer.build_analyzer()(text))
        vocabulary = vectorizer.vocabulary_
        weights = classifier.coef_[0]
        ranked = [(token, float(weights[vocabulary[token]]))
                  for token in tokens if token in vocabulary]
        top_signals = [
            {"term": token, "direction": "fraud" if weight > 0 else "legitimate",
             "weight": round(weight, 5)}
            for token, weight in sorted(ranked, key=lambda pair: abs(pair[1]), reverse=True)[:5]
        ]
    return {
        "probability": probability,
        "threshold": threshold,
        "verdict": "LIKELY_FRAUDULENT" if probability >= threshold else "LIKELY_LEGITIMATE",
        "top_signals": top_signals,
    }
