"""Leakage-safe EMSCAD training with CV model selection and one holdout evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.base import clone
from xgboost import XGBClassifier

SEED = 42
TEXT_COLUMNS = ("title", "company_profile", "description", "requirements", "benefits", "location",
                "employment_type", "required_experience", "required_education")


def load_data(path):
    df = pd.read_csv(path).drop_duplicates().reset_index(drop=True)
    if not {"title", "description", "fraudulent"}.issubset(df.columns):
        raise ValueError("CSV must contain title, description, and fraudulent columns.")
    y = pd.to_numeric(df["fraudulent"], errors="coerce")
    valid = y.isin([0, 1])
    df, y = df.loc[valid].reset_index(drop=True), y.loc[valid].astype(int).reset_index(drop=True)
    if y.nunique() != 2:
        raise ValueError("fraudulent must contain both 0 and 1.")
    cols = [c for c in TEXT_COLUMNS if c in df.columns]
    text = df[cols].fillna("").astype(str).agg(" ".join, axis=1)
    text = (text.str.replace(r"<[^>]+>", " ", regex=True)
                 .str.replace(r"https?://\S+|www\.\S+", " ", regex=True)
                 .str.replace(r"\s+", " ", regex=True).str.strip())
    return text, y


def build_models():
    # Vectorizer and SMOTE are fitted within each fold: no vocabulary or sampling leakage.
    common = [("tfidf", TfidfVectorizer(max_features=30000, ngram_range=(1, 2),
                                         min_df=2, max_df=0.98, sublinear_tf=True)),
              ("smote", SMOTE(random_state=SEED, k_neighbors=3))]
    return {
        "Logistic Regression": Pipeline(common + [("model", LogisticRegression(
            C=2.0, max_iter=2000, solver="liblinear", random_state=SEED))]),
        "Random Forest": Pipeline(common + [("model", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
            n_jobs=-1, random_state=SEED))]),
        "XGBoost": Pipeline(common + [("model", XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.9, reg_lambda=2, objective="binary:logistic",
            eval_metric="aucpr", tree_method="hist", n_jobs=-1, random_state=SEED))]),
    }


def metrics_at(y, prob, threshold):
    pred = (prob >= threshold).astype(int)
    return {"pr_auc": float(average_precision_score(y, prob)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist()}


def choose_threshold(y, prob):
    candidates = np.unique(np.r_[np.linspace(0.05, 0.95, 181), prob])
    scored = [(float(t), metrics_at(y, prob, float(t))) for t in candidates]
    best_f1 = max(scored, key=lambda x: (x[1]["f1"], x[1]["precision"], x[0]))
    high_precision = [x for x in scored if x[1]["precision"] >= 0.90]
    if high_precision:
        best_recall = max(high_precision, key=lambda x: (x[1]["recall"], x[1]["f1"]))
        if best_recall[1]["recall"] > best_f1[1]["recall"] and best_recall[1]["f1"] >= .8 * best_f1[1]["f1"]:
            return best_recall[0], "max_recall_at_precision>=0.90"
    return best_f1[0], "max_f1"


def train(data_path, output_dir="."):
    X, y_series = load_data(data_path)
    y = y_series.to_numpy()
    if len(y) < 50 or np.bincount(y, minlength=2).min() < 10:
        raise ValueError("Need at least 50 rows and 10 examples in each class for stratified 5-fold CV.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.20, stratify=y, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    # Preflight SMOTE's minimum minority count in every CV training fold.
    for tr, _ in cv.split(X_train, y_train):
        if np.bincount(y_train[tr], minlength=2).min() <= 3:
            raise ValueError("Not enough minority examples for SMOTE(k_neighbors=3) in every CV fold.")

    summaries, candidates = {}, {}
    for name, pipeline in build_models().items():
        oof = cross_val_predict(pipeline, X_train, y_train, cv=cv,
                                method="predict_proba", n_jobs=1)[:, 1]
        threshold, rule = choose_threshold(y_train, oof)
        fold_ap = []
        for tr, valid in cv.split(X_train, y_train):
            fitted_fold = clone(pipeline).fit(X_train.iloc[tr], y_train[tr])
            fold_prob = fitted_fold.predict_proba(X_train.iloc[valid])[:, 1]
            fold_ap.append(average_precision_score(y_train[valid], fold_prob))
        oof_metrics = metrics_at(y_train, oof, threshold)
        summaries[name] = {
            "cv_pr_auc_mean": float(np.mean(fold_ap)),
            "cv_pr_auc_std": float(np.std(fold_ap, ddof=1)),
            "cv_fold_pr_auc": [float(v) for v in fold_ap],
            "cv_oof_metrics": oof_metrics,
            "threshold": threshold, "threshold_rule": rule,
        }
        candidates[name] = (pipeline, threshold)

    # Model selection is based only on CV; test is not read until after this point.
    selected = max(summaries, key=lambda n: (
        summaries[n]["cv_pr_auc_mean"],
        summaries[n]["cv_oof_metrics"]["f1"],
        summaries[n]["cv_oof_metrics"]["recall"]))
    )
    pipeline, threshold = candidates[selected]
    fitted = clone(pipeline).fit(X_train, y_train)

    # The holdout is evaluated once, after model and threshold are frozen.
    test_prob = fitted.predict_proba(X_test)[:, 1]
    test_metrics = metrics_at(y_test, test_prob, threshold)
    test_metrics.update({"n": int(len(y_test)), "positive_count": int(y_test.sum())})

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": fitted, "threshold": float(threshold),
                 "model_name": selected, "text_columns": list(TEXT_COLUMNS)},
                out / "model.joblib")
    report = {
        "dataset": str(data_path), "rows_after_deduplication": int(len(y)),
        "class_counts": {"real": int((y == 0).sum()), "fraudulent": int(y.sum())},
        "test_size": .20, "random_seed": SEED,
        "selection_metric_order": ["PR-AUC", "F1", "recall"],
        "models": summaries, "selected_model": selected,
        "selected_threshold": float(threshold),
        "threshold_rule": summaries[selected]["threshold_rule"], "test": test_metrics,
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nCV RESULTS — PR-AUC mean ± std")
    print(f"{'Model':<22} {'CV PR-AUC':>18} {'OOF F1':>10} {'OOF Recall':>12}")
    for name, item in summaries.items():
        oof = item["cv_oof_metrics"]
        print(f"{name:<22} {item['cv_pr_auc_mean']:.4f} ± {item['cv_pr_auc_std']:.4f}"
              f" {oof['f1']:>10.4f} {oof['recall']:>12.4f}")
    print(f"\nSelected: {selected} | threshold={threshold:.4f} ({summaries[selected]['threshold_rule']})")
    print("\nTEST METRICS (one final evaluation)")
    print(json.dumps(test_metrics, indent=2))
    print("\nCONFUSION MATRIX [[TN, FP], [FN, TP]]")
    print(np.asarray(test_metrics["confusion_matrix"]))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/fake_job_postings.csv")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()
    train(args.data, args.output_dir)
