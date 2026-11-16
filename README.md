# Fake Job & Internship Detector

[![CI](https://github.com/manikantakarthik30-blip/fake-job-detector/actions/workflows/ci.yml/badge.svg)](https://github.com/manikantakarthik30-blip/fake-job-detector/actions/workflows/ci.yml)

Streamlit decision-support app for screening job postings with a saved TF-IDF + Logistic Regression model. Supports one-listing checks and CSV batch scoring. A clearly labelled rule-based fallback is used if the model artifact is unavailable.

> This tool assists human review; it is not a final judgment or proof of fraud. Do not use it alone to reject applicants, accuse employers, or make legal/financial decisions. Verify employers independently.

## Screenshots

Replace these placeholders with real captures saved under docs/screenshots:
- Single listing: docs/screenshots/single-listing.png
- Batch CSV: docs/screenshots/batch-csv.png
- Model card: docs/screenshots/model-card.png

## Model card and actual metrics

Metrics below are copied from the committed metrics.json training output. Dataset: fake_job_postings.csv (EMSCAD), 17,880 rows after exact deduplication: 17,014 legitimate and 866 fraudulent. Model: Logistic Regression, 5,000 unigram TF-IDF features, SMOTE inside the training pipeline. Stratified 80/20 holdout, seed 42.

| Evaluation | Metric | Value |
|---|---|---:|
| 5-fold CV | Mean PR-AUC | 0.8803 ± 0.0203 |
| 5-fold OOF | PR-AUC | 0.8802 |
| 5-fold OOF | F1 | 0.8142 |
| 5-fold OOF | Precision | 0.9003 |
| 5-fold OOF | Recall | 0.7431 |
| Holdout test | PR-AUC | 0.9421 |
| Holdout test | F1 | 0.8793 |
| Holdout test | Precision | 0.9467 |
| Holdout test | Recall | 0.8208 |

Selected threshold: **0.7279** (max F1). Holdout: 3,576 rows, 173 positive examples. These are historical-dataset results, not current real-world guarantees. Training date is not recorded in metrics.json and is therefore not stated.

A separate 1,000-feature, threshold-0.5, single-holdout model comparison is in model_comparison_holdout.json. It is a different configuration and must not be presented as a direct comparison with the 5,000-feature inference artifact.

## Architecture

~~~mermaid
flowchart TD
 U[User] --> S[Streamlit UI]
 S --> I{Single or batch}
 I -->|Single| P[predict_one]
 I -->|CSV| V[CSV validation: 20 MB / 20k rows]
 V --> B[predict_many]
 P --> M[Cached model.joblib]
 B --> M
 M --> T[TF-IDF + Logistic Regression]
 S -->|Model unavailable| R[Rule-based fallback]
 P --> O[Probability, verdict, top indicators]
 B --> O2[Results table + downloadable CSV]
 J[metrics.json] --> C[Model card]
~~~

## Setup and usage

Python 3.11 is supported.

~~~bash
git clone https://github.com/manikantakarthik30-blip/fake-job-detector.git
cd fake-job-detector
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
streamlit run app.py
~~~

Place the trained model.joblib in the repository root next to app.py. The binary model is intentionally ignored by Git; metrics.json is committed and powers the Model Card. If the model is missing, the app remains usable in rule-based fallback mode. To use a model elsewhere, set FAKE_JOB_MODEL_PATH before launching Streamlit.

### Single listing

Enter job title, company, description, requirements and logo presence. The saved model returns a probability and threshold-based verdict. The UI shows up to five indicators: red-flag phrases, short description, missing company or requirements, absent logo, and fraud-associated model terms.

### Batch CSV

Upload a UTF-8 CSV with at least one recognized text column: title, company/company_profile, description, requirements, benefits, location, employment_type, required_experience, or required_education. Label column is not required. Limits: **20 MB** and **20,000 rows**. Empty, malformed, over-limit and unrecognized files are rejected. Downloaded output includes FAKE_PROBABILITY, PREDICTION, VERDICT, SPLIT (saved_model or rule_based_fallback), and CONFIDENCE.

### Docker

~~~bash
docker build -t fake-job-detector .
docker run --rm -p 8501:8501 -v "$PWD/model.joblib:/app/model.joblib:ro" fake-job-detector
~~~

Without a mounted model artifact, the container uses the rule-based fallback.

## Tests and CI

~~~bash
ruff check .
python -m pytest -q
~~~

GitHub Actions runs Ruff and pytest on pushes and pull requests. AppTest boots the app with a temporary fixture model and submits a listing; this does not replace a deployment test using the production artifact.

## Limitations and responsible use

- EMSCAD is historical and imbalanced; present-day scams may differ.
- The training dataset is English-language; other-language performance is unknown.
- Historical labels, sampling and class imbalance can introduce bias.
- Model probabilities are not guaranteed to be calibrated for current postings.
- A missing logo, short description or missing requirements can be legitimate.
- Rule-based fallback scores are heuristic scores, not probabilities learned from data.
- A human must verify employers. Do not automate hiring rejection or fraud accusations using this output.

## Training

Obtain the EMSCAD CSV separately (do not commit raw data), place it at data/fake_job_postings.csv, then run:

~~~bash
python train.py --data data/fake_job_postings.csv --output-dir .
~~~

This writes model.joblib and metrics.json. The app does not retrain on page load.
