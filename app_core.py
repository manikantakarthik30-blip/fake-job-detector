"""CSV validation and transparent rule-based fallback."""
from __future__ import annotations
import io
import re
import pandas as pd

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_ROWS = 20_000
TEXT_COLUMNS = ("title", "company_profile", "company", "description", "requirements",
                "benefits", "location", "employment_type", "required_experience", "required_education")
RED_FLAGS = ("registration fee", "training fee", "deposit required", "wire transfer",
             "western union", "money order", "guaranteed income", "unlimited income",
             "earn money", "make money fast", "no investment", "be your own boss",
             "pyramid scheme", "mlm", "whatsapp", "telegram", "urgent requirement",
             "100% genuine", "no qualification needed")


def clean_text(value):
    value = "" if pd.isna(value) else str(value)
    value = re.sub(r"<[^>]+>|https?://\S+|www\.\S+", " ", value.lower())
    return re.sub(r"\s+", " ", value).strip()


def validate_csv(data):
    if not data:
        raise ValueError("The uploaded CSV is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("CSV exceeds the 20 MB upload limit.")
    try:
        df = pd.read_csv(io.BytesIO(data), nrows=MAX_ROWS + 1, on_bad_lines="error")
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The file has no CSV header or data.") from exc
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError("Invalid UTF-8 CSV or malformed rows. Check delimiter and encoding.") from exc
    if len(df) > MAX_ROWS:
        raise ValueError("CSV exceeds the 20,000-row limit.")
    df.columns = [str(c).strip().lower() for c in df.columns]
    if not set(df.columns).intersection(TEXT_COLUMNS):
        raise ValueError("CSV needs a job-text column: title, company, description, or requirements.")
    if df.empty:
        raise ValueError("The CSV contains a header but no listings.")
    columns = [c for c in TEXT_COLUMNS if c in df.columns]
    for col in columns:
        df[col] = df[col].fillna("").astype(str)
    df = df.loc[df[columns].apply(lambda s: s.str.strip().ne("")).any(axis=1)].reset_index(drop=True)
    if df.empty:
        raise ValueError("No usable listings found; all job-text fields are blank.")
    return df


def rule_based_prediction(record):
    title = clean_text(record.get("title", ""))
    company = clean_text(record.get("company_profile", record.get("company", "")))
    description = clean_text(record.get("description", ""))
    requirements = clean_text(record.get("requirements", ""))
    text = " ".join((title, company, description, requirements))
    flags = [flag for flag in RED_FLAGS if flag in text]
    reasons = [f"Red-flag phrase: {flag}" for flag in flags]
    words = len(description.split())
    if description and words < 40:
        reasons.append(f"Very short description ({words} words)")
    if not company:
        reasons.append("Company information is missing")
    if not requirements:
        reasons.append("Requirements are missing")
    if str(record.get("has_company_logo", record.get("logo_present", 1))).lower() in {"0", "0.0", "false", "no", "n"}:
        reasons.append("Company logo is not provided")
    score = min(95, len(flags) * 15 + (15 if description and words < 40 else 0)
                + (10 if not company else 0) + (5 if not requirements else 0)
                + (10 if "Company logo is not provided" in reasons else 0))
    return {"probability": score / 100,
            "verdict": "LIKELY_FRAUDULENT" if score >= 40 else "LIKELY_LEGITIMATE",
            "reasons": reasons[:5] or ["No configured warning indicator was found"]}
