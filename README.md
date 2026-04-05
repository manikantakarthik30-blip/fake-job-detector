# 🛡️ Fake Job & Internship Detector

AI-powered web app to detect fraudulent job/internship listings using Machine Learning.

## 🚀 Live Demo
Deploy on Streamlit Cloud → **100% FREE**

## 🤖 Algorithms Used
- Random Forest
- XGBoost
- Logistic Regression
- Gradient Boosting
- TF-IDF NLP
- SMOTE Balancing

## 📋 How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🌐 Deploy on Streamlit Cloud (Free)
1. Push this folder to GitHub
2. Go to share.streamlit.io
3. Connect your GitHub repo
4. Set main file: `app.py`
5. Click Deploy!

## 📁 Required CSV Columns
| Column | Required | Description |
|--------|----------|-------------|
| title | ✅ | Job title |
| description | ✅ | Full job description |
| requirements | Optional | Job requirements |
| company_profile | Optional | Company info |
| fraudulent | Optional | 0=Real, 1=Fake (for ML training) |
