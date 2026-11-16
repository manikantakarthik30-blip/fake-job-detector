"""Inference helpers for the saved TF-IDF classifier."""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import re
import joblib
import pandas as pd


@lru_cache(maxsize=2)
def _load_model(path):
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"Model artifact not found: {file}")
    artifact = joblib.load(file)
    _validate(artifact)
    return artifact


def _validate(artifact):
    if not isinstance(artifact, dict) or "pipeline" not in artifact or "threshold" not in artifact:
        raise ValueError("Invalid model artifact: pipeline and threshold are required.")
    if not 0 <= float(artifact["threshold"]) <= 1:
        raise ValueError("Invalid model threshold.")


def _combine(record):
    fields = ("title", "company_profile", "company", "description", "requirements",
              "benefits", "location", "employment_type", "required_experience", "required_education")
    text = " ".join(str(record.get(k) or "") for k in fields).lower()
    text = re.sub(r"<[^>]+>|https?://\S+|www\.\S+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def predict_many(records, model_path="model.joblib", model_artifact=None):
    artifact = model_artifact or _load_model(str(Path(model_path).resolve()))
    _validate(artifact)
    texts = [_combine(row) for row in records]
    if not texts or any(not text for text in texts):
        raise ValueError("Each listing must contain non-empty job text.")
    probs = artifact["pipeline"].predict_proba(pd.Series(texts))[:, 1]
    return [float(p) for p in probs]


def predict_one(record, model_path="model.joblib", model_artifact=None):
    text = _combine(record)
    if not text:
        raise ValueError("Enter a job title, company, description, or requirements.")
    artifact = model_artifact or _load_model(str(Path(model_path).resolve()))
    probability = predict_many([record], model_artifact=artifact)[0]
    threshold = float(artifact["threshold"])
    pipe = artifact["pipeline"]
    vectorizer, classifier = pipe.named_steps.get("tfidf"), pipe.named_steps.get("model")
    signals = []
    if vectorizer is not None and classifier is not None and hasattr(classifier, "coef_"):
        vocab, weights = vectorizer.vocabulary_, classifier.coef_[0]
        tokens = set(vectorizer.build_analyzer()(text))
        ranked = [(token, float(weights[vocab[token]])) for token in tokens if token in vocab]
        signals = [{"term": token, "direction": "fraud" if weight > 0 else "legitimate",
                    "weight": round(weight, 5)}
                   for token, weight in sorted(ranked, key=lambda x: abs(x[1]), reverse=True)[:5]]
    return {"probability": probability, "threshold": threshold,
            "verdict": "LIKELY_FRAUDULENT" if probability >= threshold else "LIKELY_LEGITIMATE",
            "top_signals": signals}
