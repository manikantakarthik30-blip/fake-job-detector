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

The current measured baseline is Logistic Regression with TF-IDF + SMOTE. Results are recorded in `training_metrics.json`. The holdout is a random split of this historical dataset; these scores are not evidence of live-world performance or protection against dataset/source bias. A separate Random Forest/XGBoost comparison has not been completed.

## Inference

`predict.py` exposes `predict_one({...}, model_path="model.joblib")`, returning a fraud probability, thresholded verdict, and (for Logistic Regression) the highest-weighted matched TF-IDF terms.

## Tests and CI

Run `python -m pytest -q`. GitHub Actions runs the test suite on pushes and pull requests.

## Dataset

The public EMSCAD dataset is available through Kaggle: [Real or Fake Fake Job Posting Prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction). Do not commit the raw dataset or generated model artifact to the repository. The trained `model.joblib` is generated locally by the training command.

## Limitations

This is a historical, English-language dataset. Predictions are risk signals, not proof that a job posting is fraudulent. Verify employers independently before making payments or sharing sensitive information.
