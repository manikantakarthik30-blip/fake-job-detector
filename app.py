"""
=============================================================
  🛡️ FAKE JOB & INTERNSHIP DETECTOR — Streamlit Web App
  Deploy for FREE on Streamlit Cloud
=============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, roc_auc_score, confusion_matrix)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# Visualization
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from matplotlib.gridspec import GridSpec

# ─────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🛡️ Fake Job Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  CUSTOM CSS — Dark Cyberpunk Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Share Tech Mono', monospace !important;
    background-color: #0D1117 !important;
    color: #E6EDF3 !important;
}

/* ── Hide Streamlit branding ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Main container ── */
.main .block-container {
    padding: 2rem 3rem;
    max-width: 1200px;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #161B22 !important;
    border-right: 1px solid #30363D;
}
section[data-testid="stSidebar"] * { color: #E6EDF3 !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #161B22 !important;
    border: 1px solid #30363D !important;
    border-radius: 12px !important;
    padding: 16px !important;
}
[data-testid="stMetricValue"] { color: #00D9FF !important; font-family: 'Orbitron', monospace !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #00D9FF22, #00D9FF11) !important;
    border: 1px solid #00D9FF !important;
    color: #00D9FF !important;
    border-radius: 8px !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.9rem !important;
    padding: 10px 24px !important;
    transition: all 0.3s !important;
}
.stButton > button:hover {
    background: #00D9FF22 !important;
    box-shadow: 0 0 12px rgba(0,217,255,0.3) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #161B22 !important;
    border: 2px dashed #30363D !important;
    border-radius: 12px !important;
    padding: 20px !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border: 1px solid #30363D !important; border-radius: 10px !important; }

/* ── Text input ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #161B22 !important;
    border: 1px solid #30363D !important;
    color: #E6EDF3 !important;
    border-radius: 8px !important;
    font-family: 'Share Tech Mono', monospace !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background: #161B22; border-radius: 10px; }
.stTabs [data-baseweb="tab"] { color: #8B949E !important; font-family: 'Share Tech Mono', monospace !important; }
.stTabs [aria-selected="true"] { color: #00D9FF !important; border-bottom: 2px solid #00D9FF !important; }

/* ── Progress bar ── */
.stProgress > div > div { background: linear-gradient(90deg, #00D9FF, #BD93F9) !important; }

/* ── Alerts ── */
.stSuccess { background: rgba(46,213,115,0.1) !important; border: 1px solid #2ED573 !important; border-radius: 8px !important; }
.stError   { background: rgba(255,71,87,0.1)  !important; border: 1px solid #FF4757 !important; border-radius: 8px !important; }
.stWarning { background: rgba(255,165,2,0.1)  !important; border: 1px solid #FFA502 !important; border-radius: 8px !important; }
.stInfo    { background: rgba(0,217,255,0.1)  !important; border: 1px solid #00D9FF !important; border-radius: 8px !important; }

/* ── Select box ── */
[data-baseweb="select"] > div {
    background: #161B22 !important;
    border-color: #30363D !important;
    color: #E6EDF3 !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  COLOR PALETTE for charts
# ─────────────────────────────────────────────
C = {
    'bg':      '#0D1117',
    'card':    '#161B22',
    'border':  '#30363D',
    'cyan':    '#00D9FF',
    'red':     '#FF4757',
    'green':   '#2ED573',
    'orange':  '#FFA502',
    'purple':  '#BD93F9',
    'text':    '#E6EDF3',
    'muted':   '#8B949E',
}

plt.rcParams.update({
    'figure.facecolor': C['bg'],
    'axes.facecolor':   C['card'],
    'axes.edgecolor':   C['border'],
    'axes.labelcolor':  C['text'],
    'xtick.color':      C['muted'],
    'ytick.color':      C['muted'],
    'text.color':       C['text'],
    'grid.color':       C['border'],
    'grid.alpha':       0.4,
    'font.family':      'monospace',
})

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
RED_FLAGS = [
    'earn money', 'work from home', 'no experience', 'unlimited income',
    'make money fast', 'click here', 'wire transfer', 'western union',
    'money order', 'no investment', 'be your own boss', 'residual income',
    'mlm', 'pyramid', 'guaranteed income', 'whatsapp', 'telegram',
    'extra income', 'registration fee', 'training fee', 'deposit required',
    'urgent requirement', '100% genuine', 'no qualification needed',
    'part time', 'work at home', 'home based'
]

# ─────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────

def clean_text(text):
    """Remove HTML, URLs, punctuation; lowercase."""
    if pd.isna(text):
        return ''
    text = str(text).lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def detect_target(df):
    """Auto-detect the label/fraud column."""
    for c in ['fraudulent', 'fake', 'is_fake', 'label', 'fraud', 'is_fraud', 'target']:
        if c in df.columns:
            return c
    return None


def engineer_features(df):
    """Create all ML features from raw dataframe."""
    text_cols = ['title', 'company_profile', 'description',
                 'requirements', 'benefits', 'location']
    avail = [c for c in text_cols if c in df.columns]
    df['_text'] = df[avail].fillna('').agg(' '.join, axis=1).apply(clean_text)

    df['feat_len']        = df['_text'].apply(len)
    df['feat_words']      = df['_text'].apply(lambda x: len(x.split()))
    df['feat_exclaim']    = df['_text'].apply(lambda x: x.count('!'))
    df['feat_digit_ratio']= df['_text'].apply(lambda x: sum(c.isdigit() for c in x)/max(len(x),1))
    df['feat_unique_ratio']= df['_text'].apply(lambda x: len(set(x.split()))/max(len(x.split()),1))
    df['feat_red_flags']  = df['_text'].apply(lambda x: sum(f in x for f in RED_FLAGS))

    # Binary flags
    for col in ['telecommuting', 'has_company_logo', 'has_questions']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        else:
            df[col] = 0

    # Encode categoricals
    for col in ['employment_type', 'required_experience', 'required_education']:
        if col in df.columns:
            df[col + '_enc'] = LabelEncoder().fit_transform(df[col].astype(str).fillna('Unknown'))

    return df


def build_X(df):
    """Assemble feature matrix: structured + TF-IDF."""
    struct_cols = ([c for c in df.columns if c.startswith('feat_')] +
                   ['telecommuting', 'has_company_logo', 'has_questions'] +
                   [c for c in df.columns if c.endswith('_enc')])
    struct_cols = [c for c in struct_cols if c in df.columns]

    tfidf = TfidfVectorizer(max_features=300, ngram_range=(1, 2),
                            min_df=2, sublinear_tf=True)
    X_text = tfidf.fit_transform(df['_text']).toarray()

    X_struct = df[struct_cols].fillna(0).values
    X = np.hstack([X_struct, X_text])
    X = SimpleImputer(strategy='mean').fit_transform(X)
    feat_names = struct_cols + [f'tfidf_{i}' for i in range(X_text.shape[1])]
    return X, feat_names, tfidf


@st.cache_data(show_spinner=False)
def run_pipeline(file_bytes):
    """Full ML pipeline — cached so it doesn't re-run on widget interaction."""
    df = pd.read_csv(io.BytesIO(file_bytes))
    target = detect_target(df)
    df = engineer_features(df)
    X, feat_names, tfidf = build_X(df)

    if target:
        y = pd.to_numeric(df[target], errors='coerce').fillna(0).astype(int).values

        # Balance with SMOTE
        try:
            sm = SMOTE(random_state=42, k_neighbors=min(5, max(y.sum()-1, 1)))
            X_res, y_res = sm.fit_resample(X, y)
        except Exception:
            X_res, y_res = X, y

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)

        # Train 4 models
        scale = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        models = {
            'Random Forest':      RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight='balanced'),
            'XGBoost':            xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, scale_pos_weight=scale, random_state=42, use_label_encoder=False, eval_metric='logloss', verbosity=0),
            'Logistic Regression':LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
            'Gradient Boosting':  GradientBoostingClassifier(n_estimators=150, max_depth=5, random_state=42),
        }

        results = {}
        for name, m in models.items():
            m.fit(X_tr, y_tr)
            yp = m.predict(X_te)
            results[name] = {
                'model':     m,
                'accuracy':  accuracy_score(y_te, yp),
                'f1':        f1_score(y_te, yp, zero_division=0),
                'precision': precision_score(y_te, yp, zero_division=0),
                'recall':    recall_score(y_te, yp, zero_division=0),
                'roc_auc':   roc_auc_score(y_te, m.predict_proba(X_te)[:, 1]),
                'preds':     yp,
                'proba':     m.predict_proba(X_te)[:, 1],
                'cm':        confusion_matrix(y_te, yp),
            }

        best = max(results, key=lambda k: results[k]['f1'])
        best_model = results[best]['model']

        df['PREDICTION']       = best_model.predict(X)
        df['FAKE_PROBABILITY'] = (best_model.predict_proba(X)[:, 1] * 100).round(1)
        df['VERDICT']          = df['PREDICTION'].map({0: '✅ REAL', 1: '🚨 FAKE'})

        return df, results, best, feat_names, target, True

    else:
        # Rule-based fallback
        scores = (
            df['feat_red_flags'] * 20 +
            df['feat_exclaim'].clip(0, 5) * 8 +
            (df['feat_len'] < 200).astype(int) * 30 +
            (df['has_company_logo'] == 0).astype(int) * 20
        ).clip(0, 100)

        df['FAKE_PROBABILITY'] = scores.round(1)
        df['PREDICTION']       = (scores >= 40).astype(int)
        df['VERDICT']          = df['PREDICTION'].map({0: '✅ REAL', 1: '🚨 FAKE'})

        return df, None, None, feat_names, target, False


def make_sample_csv():
    """Generate a demo CSV for download."""
    np.random.seed(42)
    n = 300
    real_titles = ['Software Engineer','Data Analyst','Product Manager',
                   'UX Designer','DevOps Engineer','Marketing Manager',
                   'Business Analyst','Financial Analyst','HR Manager']
    fake_titles = ['Work From Home Agent','Earn Money Online','Reseller Opportunity',
                   'Home-Based Data Entry','MLM Team Leader','Unlimited Income Agent',
                   'Part-Time Earner','Telegram Recruiter','Quick Money Job']
    real_desc = ('We are looking for an experienced professional to join our growing team. '
                 'The candidate should have strong technical skills and excellent communication. '
                 'Responsibilities include team collaboration, project delivery, and reporting. '
                 'We offer competitive salary, health benefits, and career growth opportunities.')
    fake_desc = ('EARN MONEY FROM HOME!! No experience needed!!!! '
                 'Work from home and make unlimited income!!! '
                 'Contact via WhatsApp or Telegram NOW!!! Registration fee required. '
                 'Be your own boss!! MLM opportunity!! 100% genuine!!!')
    labels = np.random.choice([0,1], n, p=[0.8,0.2])
    titles = [np.random.choice(fake_titles) if l else np.random.choice(real_titles) for l in labels]
    descs  = [fake_desc if l else real_desc for l in labels]
    df = pd.DataFrame({
        'title': titles, 'description': descs,
        'company_profile': np.random.choice(['Acme Corp','TechIndia','QuickCash','GlobalIT'], n),
        'location': np.random.choice(['Mumbai','Delhi','Bangalore','Remote','Chennai'], n),
        'employment_type': np.random.choice(['Full-time','Part-time','Contract'], n),
        'required_experience': np.random.choice(['0-1 years','1-3 years','3-5 years'], n),
        'telecommuting': np.random.choice([0,1], n),
        'has_company_logo': np.random.choice([0,1], n, p=[0.3,0.7]),
        'has_questions': np.random.choice([0,1], n),
        'fraudulent': labels,
    })
    return df.to_csv(index=False).encode()


# ═════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:16px 0 24px;'>
      <div style='font-family:Orbitron,monospace;font-size:1.3rem;
                  background:linear-gradient(90deg,#00D9FF,#BD93F9);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                  font-weight:900;'>🛡️ FAKE JOB</div>
      <div style='font-family:Orbitron,monospace;font-size:1.3rem;
                  background:linear-gradient(90deg,#BD93F9,#FF4757);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                  font-weight:900;'>DETECTOR</div>
      <div style='color:#8B949E;font-size:0.7rem;margin-top:6px;letter-spacing:2px;'>
        AI FRAUD DETECTION v1.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("**📋 REQUIRED CSV COLUMNS**")
    st.markdown("""
    <div style='font-size:0.78rem;color:#8B949E;line-height:1.9;'>
    • <code style='color:#BD93F9;'>title</code> — Job title<br>
    • <code style='color:#BD93F9;'>description</code> — Full description<br>
    • <code style='color:#BD93F9;'>requirements</code> — Requirements<br>
    • <code style='color:#BD93F9;'>company_profile</code> — Company info<br>
    • <code style='color:#2ED573;'>fraudulent</code> — 0=Real, 1=Fake<br>
    <span style='color:#FFA502;font-size:0.72rem;'>*(fraudulent column is optional)*</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**🤖 ALGORITHMS**")
    st.markdown("""
    <div style='font-size:0.78rem;color:#8B949E;line-height:1.9;'>
    • Random Forest<br>• XGBoost<br>• Logistic Regression<br>
    • Gradient Boosting<br>• TF-IDF NLP<br>• SMOTE Balancing
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**📥 SAMPLE DATASET**")
    st.download_button(
        label="⬇️ Download Sample CSV",
        data=make_sample_csv(),
        file_name="sample_jobs.csv",
        mime="text/csv"
    )

    st.markdown("---")
    st.markdown("""
    <div style='color:#8B949E;font-size:0.72rem;text-align:center;'>
    Built with Streamlit + sklearn + XGBoost<br>
    Free hosting on Streamlit Cloud
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════
#  MAIN HEADER
# ═════════════════════════════════════════════
st.markdown("""
<div style='background:linear-gradient(135deg,#161B22,#0D1117);
            border:1px solid #30363D;border-radius:16px;
            padding:36px 40px;margin-bottom:28px;position:relative;overflow:hidden;'>
  <div style='position:absolute;top:0;left:0;right:0;bottom:0;
              background:radial-gradient(circle at 20% 50%,rgba(0,217,255,0.06) 0%,transparent 60%),
                          radial-gradient(circle at 80% 50%,rgba(255,71,87,0.06) 0%,transparent 60%);
              pointer-events:none;'></div>
  <div style='font-family:Orbitron,monospace;font-size:2rem;font-weight:900;
              background:linear-gradient(90deg,#00D9FF,#BD93F9,#FF4757);
              -webkit-background-clip:text;-webkit-text-fill-color:transparent;
              background-clip:text;letter-spacing:3px;'>
    🛡️ FAKE JOB & INTERNSHIP DETECTOR
  </div>
  <div style='color:#8B949E;font-size:0.88rem;margin-top:8px;letter-spacing:2px;'>
    AI-POWERED FRAUD DETECTION SYSTEM &nbsp;|&nbsp; UPLOAD → ANALYZE → PROTECT
  </div>
  <div style='margin-top:16px;'>
    <span style='background:rgba(0,217,255,0.15);color:#00D9FF;border:1px solid rgba(0,217,255,0.3);
                 border-radius:20px;padding:3px 12px;font-size:0.72rem;margin:3px;display:inline-block;'>
      Random Forest</span>
    <span style='background:rgba(189,147,249,0.15);color:#BD93F9;border:1px solid rgba(189,147,249,0.3);
                 border-radius:20px;padding:3px 12px;font-size:0.72rem;margin:3px;display:inline-block;'>
      XGBoost</span>
    <span style='background:rgba(46,213,115,0.15);color:#2ED573;border:1px solid rgba(46,213,115,0.3);
                 border-radius:20px;padding:3px 12px;font-size:0.72rem;margin:3px;display:inline-block;'>
      TF-IDF NLP</span>
    <span style='background:rgba(255,71,87,0.15);color:#FF4757;border:1px solid rgba(255,71,87,0.3);
                 border-radius:20px;padding:3px 12px;font-size:0.72rem;margin:3px;display:inline-block;'>
      SMOTE Balancing</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════
#  TABS
# ═════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["📂  BATCH ANALYSIS", "🔎  SINGLE CHECK", "📖  HOW IT WORKS"])


# ───────────────────────────────────────────
#  TAB 1 — BATCH ANALYSIS (CSV Upload)
# ───────────────────────────────────────────
with tab1:
    st.markdown("#### Upload your CSV dataset")

    uploaded = st.file_uploader(
        "Drop your CSV here — or use the sample from the sidebar",
        type=['csv'],
        help="CSV with job/internship listings. Add a 'fraudulent' column (0/1) for full ML training."
    )

    if uploaded:
        file_bytes = uploaded.read()

        with st.spinner("🤖 Running AI analysis... (30–90 sec for large files)"):
            df_out, ml_results, best_name, feat_names, target_col, supervised = run_pipeline(file_bytes)

        # ── SUMMARY METRICS ──
        total     = len(df_out)
        fake_cnt  = (df_out['PREDICTION'] == 1).sum()
        real_cnt  = (df_out['PREDICTION'] == 0).sum()
        high_risk = (df_out['FAKE_PROBABILITY'] >= 70).sum()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📋 Total Listings",  f"{total:,}")
        c2.metric("✅ Real Jobs",        f"{real_cnt:,}")
        c3.metric("🚨 Fake Jobs",        f"{fake_cnt:,}",   f"{fake_cnt/total*100:.1f}% of total")
        c4.metric("⚠️ High Risk (>70%)", f"{high_risk:,}")

        # ── BEST MODEL BADGE ──
        if supervised and ml_results:
            bd = ml_results[best_name]
            st.success(f"🏆 Best Model: **{best_name}** — F1: `{bd['f1']:.3f}` | AUC: `{bd['roc_auc']:.3f}` | Accuracy: `{bd['accuracy']:.3f}`")

            # ── MODEL LEADERBOARD ──
            with st.expander("📊 Model Leaderboard — click to expand", expanded=True):
                rows = []
                medals = ['🥇','🥈','🥉','🏅']
                for i, (nm, d) in enumerate(sorted(ml_results.items(), key=lambda x: x[1]['f1'], reverse=True)):
                    rows.append({
                        'Rank': medals[i] if i < 4 else '',
                        'Model': nm,
                        'Accuracy': f"{d['accuracy']:.3f}",
                        'F1 Score': f"{d['f1']:.3f}",
                        'Precision': f"{d['precision']:.3f}",
                        'Recall': f"{d['recall']:.3f}",
                        'AUC-ROC': f"{d['roc_auc']:.3f}",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # ── CHARTS ──
            with st.expander("📈 Visual Analysis Charts", expanded=True):
                chart_col1, chart_col2 = st.columns(2)

                # Chart 1: Class distribution
                with chart_col1:
                    fig, ax = plt.subplots(figsize=(5, 3.5), facecolor=C['bg'])
                    if target_col and target_col in df_out.columns:
                        counts = df_out[target_col].value_counts()
                        labels_map = {0: 'REAL', 1: 'FAKE'}
                        bar_colors = [C['green'] if k==0 else C['red'] for k in counts.index]
                        ax.bar([labels_map.get(k,str(k)) for k in counts.index],
                               counts.values, color=bar_colors, width=0.5,
                               edgecolor=C['border'], linewidth=1.5)
                    ax.set_title('Dataset Distribution', color=C['cyan'], pad=10)
                    ax.grid(axis='y', alpha=0.3)
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

                # Chart 2: Red flag distribution
                with chart_col2:
                    fig, ax = plt.subplots(figsize=(5, 3.5), facecolor=C['bg'])
                    if target_col and target_col in df_out.columns:
                        real_flags = df_out[df_out[target_col]==0]['feat_red_flags']
                        fake_flags = df_out[df_out[target_col]==1]['feat_red_flags']
                        ax.hist(real_flags, bins=15, alpha=0.8, color=C['green'], label='REAL')
                        ax.hist(fake_flags, bins=15, alpha=0.8, color=C['red'],   label='FAKE')
                        ax.legend(facecolor=C['card'], edgecolor=C['border'], labelcolor=C['text'])
                    ax.set_title('Red Flag Word Distribution', color=C['cyan'], pad=10)
                    ax.set_xlabel('Red Flag Count')
                    ax.grid(alpha=0.3)
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

                # Chart 3: Model comparison bar chart
                fig, ax = plt.subplots(figsize=(11, 4), facecolor=C['bg'])
                metrics = ['accuracy','f1','precision','recall','roc_auc']
                names   = list(ml_results.keys())
                x = np.arange(len(metrics))
                w = 0.18
                palette = [C['cyan'], C['red'], C['green'], C['purple']]
                for i, (nm, d) in enumerate(ml_results.items()):
                    ax.bar(x + i*w, [d[m] for m in metrics], w,
                           label=nm, color=palette[i%4], alpha=0.85,
                           edgecolor=C['bg'], linewidth=0.5)
                ax.set_xticks(x + w*(len(names)-1)/2)
                ax.set_xticklabels([m.upper().replace('_',' ') for m in metrics])
                ax.set_ylim(0, 1.15)
                ax.legend(facecolor=C['card'], edgecolor=C['border'],
                          labelcolor=C['text'], loc='upper right', fontsize=8)
                ax.set_title('Model Performance Comparison', color=C['cyan'], pad=10)
                ax.grid(axis='y', alpha=0.3)
                st.pyplot(fig, use_container_width=True)
                plt.close()

                # Chart 4: Confusion matrices
                n = len(ml_results)
                fig, axes = plt.subplots(1, n, figsize=(4.5*n, 4), facecolor=C['bg'])
                if n == 1: axes = [axes]
                cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                    'c', [C['bg'], C['cyan']], N=256)
                for ax, (nm, d) in zip(axes, ml_results.items()):
                    sns.heatmap(d['cm'], annot=True, fmt='d', cmap=cmap, ax=ax,
                                linewidths=1.5, linecolor=C['bg'], cbar=False,
                                annot_kws={'size':13,'weight':'bold','color':C['text']})
                    ax.set_title(f'{nm}\nF1={d["f1"]:.3f}', color=C['cyan'], fontsize=9)
                    ax.set_xticklabels(['REAL','FAKE'], color=C['text'])
                    ax.set_yticklabels(['REAL','FAKE'], color=C['text'], rotation=0)
                    ax.set_xlabel('Predicted', color=C['muted'])
                    ax.set_ylabel('Actual',    color=C['muted'])
                plt.suptitle('Confusion Matrices', color=C['text'], fontsize=12, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

                # Chart 5: Feature importance (Random Forest)
                if 'Random Forest' in ml_results:
                    rf_model = ml_results['Random Forest']['model']
                    if hasattr(rf_model, 'feature_importances_'):
                        imp = rf_model.feature_importances_
                        top_idx = np.argsort(imp)[-15:]
                        top_names = [feat_names[i] if i < len(feat_names) else f'f{i}' for i in top_idx]
                        top_vals  = imp[top_idx]
                        fig, ax = plt.subplots(figsize=(9, 5), facecolor=C['bg'])
                        colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(top_names)))
                        ax.barh(top_names, top_vals, color=colors,
                                edgecolor=C['border'], linewidth=0.8)
                        ax.set_title('Top Features for Fake Job Detection (Random Forest)',
                                     color=C['cyan'], pad=10)
                        ax.set_xlabel('Importance Score', color=C['muted'])
                        ax.grid(axis='x', alpha=0.3)
                        st.pyplot(fig, use_container_width=True)
                        plt.close()

        else:
            st.info("ℹ️ No `fraudulent` label column found — showing rule-based predictions.")

        # ── PREDICTION TABLE ──
        st.markdown("#### 📋 Prediction Results")
        display_cols = ['VERDICT', 'FAKE_PROBABILITY', 'feat_red_flags']
        if 'title' in df_out.columns:        display_cols = ['title']       + display_cols
        if 'location' in df_out.columns:     display_cols = display_cols    + ['location']
        if 'employment_type' in df_out.columns: display_cols += ['employment_type']

        show_df = df_out[display_cols].copy()
        show_df.columns = [c.replace('feat_red_flags','🚩 Red Flags')
                            .replace('FAKE_PROBABILITY','Fake % ')
                            .replace('VERDICT','Verdict')
                            .replace('title','Job Title')
                            .replace('location','Location')
                            .replace('employment_type','Type')
                           for c in show_df.columns]

        st.dataframe(show_df, use_container_width=True, height=420)

        # ── EXPORT ──
        st.markdown("#### 💾 Download Results")
        csv_out = df_out.to_csv(index=False).encode()
        st.download_button(
            label="⬇️ Download Full Results CSV",
            data=csv_out,
            file_name="fake_job_results.csv",
            mime="text/csv"
        )


# ───────────────────────────────────────────
#  TAB 2 — SINGLE LISTING CHECK
# ───────────────────────────────────────────
with tab2:
    st.markdown("#### 🔎 Check a Single Job Listing Instantly")
    st.markdown("<div style='color:#8B949E;font-size:0.85rem;'>Paste any job listing below — get an instant AI verdict.</div>", unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 1])
    with col_a:
        job_title = st.text_input("Job Title", placeholder="e.g. Software Engineer at TechCorp")
    with col_b:
        job_company = st.text_input("Company Name", placeholder="e.g. Acme Technologies")

    job_desc = st.text_area(
        "Job Description / Requirements",
        placeholder="Paste the full job description here...",
        height=180
    )

    col_btn1, col_btn2 = st.columns([1, 5])
    with col_btn1:
        check_clicked = st.button("🚨 CHECK NOW", use_container_width=True)

    if check_clicked:
        if not job_title.strip() and not job_desc.strip():
            st.warning("⚠️ Please enter a job title or description.")
        else:
            combined = clean_text(f"{job_title} {job_company} {job_desc}")
            found_flags   = [f for f in RED_FLAGS if f in combined]
            exclaim_count = (job_title + job_desc).count('!')
            is_short      = len(job_desc.split()) < 40
            all_caps_pct  = sum(1 for c in job_title+job_desc if c.isupper()) / max(len(job_title+job_desc), 1)

            score = min(
                len(found_flags) * 18 +
                min(exclaim_count * 5, 25) +
                (20 if is_short and job_desc else 0) +
                (15 if all_caps_pct > 0.2 else 0),
                100
            )

            if score >= 60:
                verdict, vcolor, icon = "🚨 LIKELY FAKE", "#FF4757", "🔴"
            elif score >= 30:
                verdict, vcolor, icon = "⚠️ SUSPICIOUS",  "#FFA502", "🟡"
            else:
                verdict, vcolor, icon = "✅ LIKELY REAL",  "#2ED573", "🟢"

            # Result card
            st.markdown(f"""
            <div style='background:#161B22;border:2px solid {vcolor};border-radius:14px;
                        padding:24px 28px;margin:16px 0;'>
              <div style='font-size:1.6rem;font-weight:bold;color:{vcolor};margin-bottom:16px;'>
                {icon} {verdict}
              </div>
              <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-bottom:16px;'>
                <div style='text-align:center;'>
                  <div style='font-size:2.2rem;font-weight:bold;color:{vcolor};'>{score}</div>
                  <div style='color:#8B949E;font-size:0.8rem;'>RISK SCORE / 100</div>
                </div>
                <div style='text-align:center;'>
                  <div style='font-size:2.2rem;font-weight:bold;color:#FF4757;'>{len(found_flags)}</div>
                  <div style='color:#8B949E;font-size:0.8rem;'>RED FLAGS FOUND</div>
                </div>
                <div style='text-align:center;'>
                  <div style='font-size:2.2rem;font-weight:bold;color:#FFA502;'>{exclaim_count}</div>
                  <div style='color:#8B949E;font-size:0.8rem;'>EXCLAMATION MARKS</div>
                </div>
              </div>
              <div style='background:#0D1117;border-radius:6px;height:12px;overflow:hidden;margin-bottom:16px;'>
                <div style='background:{vcolor};height:100%;width:{score}%;transition:width 1s;'></div>
              </div>
              <div style='margin-top:12px;'>
                <span style='color:#8B949E;font-size:0.85rem;'>🚩 Red Flag Keywords Detected:</span><br>
                <div style='margin-top:8px;'>
            """, unsafe_allow_html=True)

            if found_flags:
                flags_html = ''.join([
                    f"<span style='background:rgba(255,71,87,0.15);color:#FF4757;"
                    f"border:1px solid rgba(255,71,87,0.3);border-radius:4px;"
                    f"padding:2px 10px;margin:3px;display:inline-block;font-size:0.8rem;'>"
                    f"🚩 {f}</span>" for f in found_flags
                ])
            else:
                flags_html = "<span style='color:#2ED573;'>None detected ✅</span>"

            st.markdown(f"""
                {flags_html}
                </div>
              </div>
              <div style='border-top:1px solid #30363D;margin-top:16px;padding-top:12px;
                          color:#8B949E;font-size:0.8rem;'>
                Short Description: {'⚠️ Yes' if is_short and job_desc else '✅ No'} &nbsp;|&nbsp;
                Excessive Caps: {'⚠️ Yes' if all_caps_pct > 0.2 else '✅ No'}
              </div>
            </div>
            """, unsafe_allow_html=True)


# ───────────────────────────────────────────
#  TAB 3 — HOW IT WORKS
# ───────────────────────────────────────────
with tab3:
    st.markdown("""
    <div style='background:#161B22;border:1px solid #30363D;border-radius:14px;padding:28px;'>

      <div style='color:#00D9FF;font-size:1.15rem;font-weight:bold;margin-bottom:20px;'>
        🔬 HOW THE DETECTOR WORKS
      </div>

      <div style='display:grid;grid-template-columns:1fr 1fr;gap:24px;'>
        <div>
          <div style='color:#FFA502;font-weight:bold;margin-bottom:10px;'>📊 ML PIPELINE STEPS</div>
          <div style='color:#8B949E;font-size:0.83rem;line-height:2;'>
            <b style='color:#00D9FF;'>1. Text Cleaning</b> — Remove HTML, URLs, noise<br>
            <b style='color:#00D9FF;'>2. Feature Engineering</b> — Extract 300+ text + numeric features<br>
            <b style='color:#00D9FF;'>3. TF-IDF Vectorization</b> — Convert text to ML-ready format<br>
            <b style='color:#00D9FF;'>4. SMOTE Balancing</b> — Fix class imbalance synthetically<br>
            <b style='color:#00D9FF;'>5. Train 4 Models</b> — RF, XGBoost, LR, GradBoost<br>
            <b style='color:#00D9FF;'>6. Ensemble Vote</b> — Best model selected by F1 score<br>
            <b style='color:#00D9FF;'>7. Predict + Score</b> — Label each listing with % fake probability
          </div>
        </div>

        <div>
          <div style='color:#FFA502;font-weight:bold;margin-bottom:10px;'>🚩 FAKE JOB RED FLAGS</div>
          <div style='color:#8B949E;font-size:0.83rem;line-height:2;'>
            • "Earn Money", "Work From Home" phrases<br>
            • Excessive exclamation marks (!!!)<br>
            • Registration / Training fees required<br>
            • Contact via WhatsApp / Telegram only<br>
            • "No experience needed" / "Guaranteed income"<br>
            • Very short, vague job descriptions<br>
            • MLM / Pyramid scheme language<br>
            • No company logo / professional profile
          </div>
        </div>
      </div>

      <div style='border-top:1px solid #30363D;margin-top:24px;padding-top:20px;
                  display:grid;grid-template-columns:1fr 1fr;gap:24px;'>
        <div>
          <div style='color:#FFA502;font-weight:bold;margin-bottom:10px;'>📁 DATASET SOURCES</div>
          <div style='color:#8B949E;font-size:0.83rem;line-height:2;'>
            • <a href='https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction'
                 style='color:#00D9FF;'>Kaggle: EMSCAD Fake Job Dataset</a><br>
            • Any CSV with job data + fraudulent (0/1) column<br>
            • Min 50 rows for ML training<br>
            • Best: 1000+ rows, balanced classes
          </div>
        </div>
        <div>
          <div style='color:#FFA502;font-weight:bold;margin-bottom:10px;'>📈 METRICS EXPLAINED</div>
          <div style='color:#8B949E;font-size:0.83rem;line-height:2;'>
            • <b style='color:#00D9FF;'>Accuracy</b> — % of correct predictions<br>
            • <b style='color:#00D9FF;'>F1 Score</b> — Balance of precision + recall<br>
            • <b style='color:#00D9FF;'>Precision</b> — Of flagged fakes, how many are real fakes<br>
            • <b style='color:#00D9FF;'>Recall</b> — Of all real fakes, how many were caught<br>
            • <b style='color:#00D9FF;'>AUC-ROC</b> — Overall model quality (1.0 = perfect)
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
