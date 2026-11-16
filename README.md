# Fake Job & Internship Detector

Streamlit application and an independently testable NLP inference module for detecting potentially fraudulent job postings.

## Training the saved model

Place the EMSCAD/Kaggle CSV at `data/fake_job_postings.csv` (or pass another path):

```bash
pip install -r requirements.txt
pip install pytest
python train.py --data data/fake_job_postings.csv --output-dir .
pytest -q
```

The training script:
- Removes exact duplicate rows and validates the binary `fraudulent` target.
- Uses a stratified 80/20 holdout split.
- Fits TF-IDF and SMOTE only inside each training fold.
- Uses five-fold stratified out-of-fold predictions to select a decision threshold.
- Fits the final pipeline on the training partition and evaluates the holdout once.
- Writes `model.joblib` and `metrics.json`.

The current production inference artifact is Logistic Regression with 5,000 TF-IDF features + SMOTE; its five-fold CV and holdout metrics are recorded in `training_metrics.json`/`metrics.json`. A separate like-for-like model comparison on a stratified 80/20 split, using 1,000 TF-IDF features and SMOTE on training data only, is recorded in `model_comparison_holdout.json`. In that comparison Random Forest achieved the best F1 (0.7554) and PR-AUC (0.8406). XGBoost reached PR-AUC 0.7640 and F1 0.6789 at threshold 0.5. All models use SMOTE on training data only; XGBoost does not receive a second imbalance weight. The comparison is a single holdout experiment, not five-fold CV. All scores are from a historical dataset and do not guarantee real-world performance.

## Inference

`predict.py` exposes `predict_one({...}, model_path="model.joblib")`, returning a fraud probability, decision threshold, thresholded verdict, and (for Logistic Regression) the highest-weighted matched TF-IDF terms. The Streamlit **SINGLE CHECK** tab calls this function; it no longer uses the old hand-written keyword score. Place the generated `model.joblib` in the repository root before launching or deploying the app. If it is absent, the app displays the exact training command instead of silently using a fallback score.

## Tests and CI

Run `python -m pytest -q`. GitHub Actions runs the test suite on pushes and pull requests.

## Dataset

The public EMSCAD dataset is available through Kaggle: [Real or Fake Fake Job Posting Prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction). Do not commit the raw dataset or generated model artifact to the repository. The trained `model.joblib` is generated locally by the training command.

## Limitations

This is a historical, English-language dataset. Predictions are risk signals, not proof that a job posting is fraudulent. Verify employers independently before making payments or sharing sensitive information.

## Streamlit single-check integration

The single-listing form sends the entered title, company name, and description to the saved model. The result displays the model probability, selected threshold, verdict, and top weighted terms. The CI suite includes a Streamlit AppTest that boots the app, submits a sample listing, and checks the rendered prediction. The batch-analysis tab also now splits raw rows before fitting TF-IDF and SMOTE, evaluates on untouched holdout rows, and shows the selected model's predictions with train/test labels.
