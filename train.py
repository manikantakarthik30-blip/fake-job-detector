"""Train a leakage-safe TF-IDF + Logistic Regression fake-job classifier."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split

SEED = 42
TEXT_COLUMNS = ("title", "company_profile", "description", "requirements", "benefits",
                "location", "employment_type", "required_experience", "required_education")


def load_data(path):
    df = pd.read_csv(path).drop_duplicates().reset_index(drop=True)
    required = {"title", "description", "fraudulent"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain: {', '.join(sorted(required))}")
    target = pd.to_numeric(df["fraudulent"], errors="coerce")
    valid = target.isin([0, 1])
    df = df.loc[valid].reset_index(drop=True)
    y = target.loc[valid].astype(int).reset_index(drop=True)
    if y.nunique() != 2:
        raise ValueError("fraudulent must contain both 0 and 1.")
    columns = [column for column in TEXT_COLUMNS if column in df.columns]
    text = df[columns].fillna("").astype(str).agg(" ".join, axis=1)
    text = (text.str.replace(r"<[^>]+>", " ", regex=True)
                .str.replace(r"https?://\S+|www\.\S+", " ", regex=True)
                .str.replace(r"\s+", " ", regex=True).str.strip())
    return text, y


def build_pipeline():
    # Vocabulary and SMOTE are fitted inside each training fold only.
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 1),
                                  min_df=2, max_df=0.98, sublinear_tf=True)),
        ("smote", SMOTE(random_state=SEED, k_neighbors=3)),
        ("model", LogisticRegression(C=2, max_iter=1000, solver="liblinear",
                                     random_state=SEED)),
    ])


def metrics_at(y_true, probabilities, threshold):
    predicted = (probabilities >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, predicted, labels=[0, 1]).tolist(),
    }


def choose_threshold(y_true, probabilities):
    """Vectorized threshold sweep; avoids quadratic repeated metric computation."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if not len(thresholds):
        return 0.5, "default_no_thresholds"
    precision, recall = precision[:-1], recall[:-1]
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best_f1 = int(np.argmax(f1))
    eligible = np.flatnonzero((precision >= 0.90) & (recall > 0))
    if len(eligible):
        best_recall = int(eligible[np.argmax(recall[eligible])])
        if recall[best_recall] > recall[best_f1] and f1[best_recall] >= 0.8 * f1[best_f1]:
            return float(thresholds[best_recall]), "max_recall_at_precision>=0.90"
    return float(thresholds[best_f1]), "max_f1"


def train(data_path, output_dir="."):
    X, y_series = load_data(data_path)
    y = y_series.to_numpy()
    if len(y) < 50 or np.bincount(y, minlength=2).min() < 10:
        raise ValueError("Need at least 50 rows and 10 examples in each class for 5-fold CV.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(cv.split(X_train, y_train))
    if any(np.bincount(y_train[tr], minlength=2).min() <= 3 for tr, _ in splits):
        raise ValueError("Not enough minority examples for SMOTE(k_neighbors=3).")

    pipeline = build_pipeline()
    oof = np.zeros(len(y_train), dtype=float)
    fold_pr_auc = []
    # One fold loop collects both out-of-fold predictions and fold PR-AUC.
    for train_idx, valid_idx in splits:
        fold_model = clone(pipeline).fit(X_train.iloc[train_idx], y_train[train_idx])
        fold_prob = fold_model.predict_proba(X_train.iloc[valid_idx])[:, 1]
        oof[valid_idx] = fold_prob
        fold_pr_auc.append(average_precision_score(y_train[valid_idx], fold_prob))

    threshold, threshold_rule = choose_threshold(y_train, oof)
    final_model = clone(pipeline).fit(X_train, y_train)
    test_prob = final_model.predict_proba(X_test)[:, 1]
    test_metrics = metrics_at(y_test, test_prob, threshold)
    test_metrics.update({"n": int(len(y_test)), "positive_count": int(y_test.sum())})

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": final_model, "threshold": float(threshold),
                 "model_name": "Logistic Regression", "text_columns": list(TEXT_COLUMNS)},
                output / "model.joblib")
    report = {
        "dataset": Path(data_path).name,
        "rows_after_deduplication": int(len(y)),
        "class_counts": {"real": int((y == 0).sum()), "fraudulent": int(y.sum())},
        "test_size": 0.20, "random_seed": SEED,
        "cross_validation": {"folds": 5, "method": "StratifiedKFold",
                             "smote_inside_pipeline": True},
        "model": "Logistic Regression",
        "cv_pr_auc_mean": float(np.mean(fold_pr_auc)),
        "cv_pr_auc_std": float(np.std(fold_pr_auc, ddof=1)),
        "cv_fold_pr_auc": [float(value) for value in fold_pr_auc],
        "cv_oof_metrics": metrics_at(y_train, oof, threshold),
        "selected_threshold": float(threshold), "threshold_rule": threshold_rule,
        "test": test_metrics,
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/fake_job_postings.csv")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()
    train(args.data, args.output_dir)
