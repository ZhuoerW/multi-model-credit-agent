"""
evaluate_model.py — Load a saved model artifact and produce analysis outputs.

Usage:
    python evaluate_model.py --model_path ../outputs/models/xgboost_relief_classifier_with_topics.joblib
                             --input_csv  ../data/raw/complaints_with_topics.csv
                             --output_dir ../outputs/results/xgboost_with_topics

Outputs written to --output_dir:
    test_predictions.csv    complaint metadata + true label + predicted prob + predicted label
    pr_curve.csv            precision / recall / threshold (for PR-curve plot)
    roc_curve.csv           FPR / TPR / threshold (for ROC-curve plot)
    feature_importance.csv  feature importance scores (XGBoost) or SVM coefficients
    metrics.json            scalar metrics + per-threshold breakdown

Works with any model saved as {"pipeline": <Pipeline>, "threshold": <float>}.
"""

import argparse
import json
import os
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class IndexMaskSelector(BaseEstimator, TransformerMixin):
    """Needed to deserialize models saved by 05_xgboost_structured_features.py."""
    def __init__(self, mask):
        self.mask = mask
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        import scipy.sparse as sp
        col_idx = np.where(self.mask)[0]
        return X[:, col_idx] if sp.issparse(X) else X[:, self.mask]
    def get_support(self):
        return self.mask
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# ── paths ─────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL  = os.path.join(_HERE, "..", "outputs", "models",
                               "xgboost_relief_classifier_with_topics.joblib")
DEFAULT_INPUT  = os.path.join(_HERE, "..", "data", "raw", "complaints_with_topics.csv")
DEFAULT_OUTPUT = os.path.join(_HERE, "..", "outputs", "results", "xgboost_with_topics")

# ── columns ───────────────────────────────────────────────────────────────────
TEXT_COL     = "Consumer complaint narrative"
RESPONSE_COL = "Company response to consumer"
TEXT_FEATURE = "clean_text"

RELIEF_RESPONSE = "Closed with monetary relief"

# Metadata columns kept in the output CSV (not model features)
META_COLS = [TEXT_COL, RESPONSE_COL, "Product", "Issue", "primary_topic", "topic_confidence"]

# ── text cleaning (must match training) ───────────────────────────────────────
_RE_REDACTION  = re.compile(r"\bx{2,}\b", re.IGNORECASE)
_RE_DIGITS     = re.compile(r"\b\d+\b")
_RE_PUNCT      = re.compile(r"[^a-z\s]")
_RE_WHITESPACE = re.compile(r"\s+")
_RE_BOILERPLATE = re.compile(
    r"(?:"
    r"in accordance with the fair credit reporting act[^.]*\."
    r"|the fair credit reporting act\s*\([^)]*\)\s*says[^.]*\."
    r"|i have not supplied proof under the doctrine of estoppel[^.]*\."
    r"|this cfpb complaint has been filed to request pursuant to fcra[^.]*\."
    r"|you have reported inaccurate and unauthorized accounts on my credit report[^.]*\."
    r")",
    re.IGNORECASE,
)


_RE_DOLLAR_ANY = re.compile(r"\{\$[\d,]+\.?\d*\}|\$[\d,]+\.?\d*")
_RE_SENTENCE   = re.compile(r"[.!?]+")
_RE_VOWEL      = re.compile(r"[aeiouy]+")


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = _RE_BOILERPLATE.sub(" ", text)
    text = _RE_REDACTION.sub(" ", text)
    text = _RE_DOLLAR_ANY.sub(" dollar_amount ", text)
    text = _RE_PUNCT.sub(" ", text)
    text = _RE_WHITESPACE.sub(" ", text).strip()
    return text


# ── narrative feature helpers (match 05_xgboost_structured_features.py) ───────

def _extract_dollar_amounts(text):
    amounts = []
    for m in re.findall(r'\$[\d,]+(?:\.\d+)?', str(text)):
        try:
            amounts.append(float(m.replace('$', '').replace(',', '')))
        except ValueError:
            pass
    return amounts

def _count_syllables(word):
    word = word.lower().strip(".,!?;:'\"()-")
    if not word:
        return 0
    count = len(_RE_VOWEL.findall(word))
    if word.endswith('e') and count > 1:
        count -= 1
    return max(1, count)

def _flesch(text):
    words = str(text).split()
    if not words:
        return np.nan
    sentences = max(1, len(re.findall(r'[.!?]+', text)))
    syllables = sum(_count_syllables(w) for w in words)
    return 206.835 - 1.015 * (len(words) / sentences) - 84.6 * (syllables / len(words))


# ── data loading ──────────────────────────────────────────────────────────────

def load_and_prepare(csv_path: str) -> pd.DataFrame:
    """
    Computes all possible features so any saved model can be evaluated.
    The pipeline's ColumnTransformer(remainder='drop') ignores columns it
    wasn't trained on, so extra columns are harmless.
    """
    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  Raw rows               : {len(df):,}")

    df = df.dropna(subset=[TEXT_COL])
    df = df[df[TEXT_COL].str.len() >= 50]
    print(f"  After narrative filter : {len(df):,}")

    df = df.dropna(subset=[RESPONSE_COL])
    print(f"  After response filter  : {len(df):,}")

    df = df.drop_duplicates(subset=[TEXT_COL])
    print(f"  After deduplication    : {len(df):,}")

    df["label"] = (df[RESPONSE_COL] == RELIEF_RESPONSE).astype(int)
    df["date"]  = pd.to_datetime(df["Date received"], errors="coerce")

    # ── narrative numeric features (for structured model) ─────────────────────
    raw = df[TEXT_COL]
    dollar_series             = raw.apply(_extract_dollar_amounts)
    df["dollar_amount_count"] = dollar_series.apply(len).astype(float)
    df["max_dollar_amount"]   = dollar_series.apply(lambda x: max(x) if x else 0.0).astype(float)
    df["total_dollar_amount"] = dollar_series.apply(lambda x: float(sum(x)) if x else 0.0)
    df["sentence_count"]      = raw.apply(lambda t: max(len([s for s in _RE_SENTENCE.split(str(t)) if s.strip()]), 1)).astype(float)
    df["flesch_kincaid"]      = raw.apply(_flesch).astype(float)
    df["lexical_diversity"]   = raw.apply(lambda t: len(set(t.lower().split())) / len(t.split()) if t.split() else 0.0).astype(float)
    df["char_len"]            = raw.str.len().astype(float)
    df["date_count"]          = raw.str.count(r'\b(?:XX|\d{1,2})/(?:XX|\d{1,2})/(?:XX|\d{2,4})\b').astype(float)
    df["caps_ratio"]          = raw.apply(lambda t: sum(1 for w in t.split() if w.isupper() and len(w) > 1) / max(len(t.split()), 1)).astype(float)
    df["avg_word_len"]        = raw.apply(lambda t: float(np.mean([len(w) for w in t.split()])) if t.split() else 0.0).astype(float)
    df["exclamation_count"]   = raw.str.count(r'!').astype(float)
    df["account_ref_count"]   = raw.str.count(r'\bX{4,}\b|\b\d{4,}\b').astype(float)
    df["duration_mention_count"]  = raw.str.count(r'\b(?:month|months|week|weeks|year|years|day|days)\b', flags=re.IGNORECASE).astype(float)
    df["contact_attempt_count"]   = raw.str.count(r'\b(?:called|emailed|email|spoke with|spoken with|visited|wrote|contacted|reached out|called back|speaking with)\b', flags=re.IGNORECASE).astype(float)
    df["escalation_person_count"] = raw.str.count(r'\b(?:representative|manager|supervisor|agent|customer service|customer care|specialist|officer)\b', flags=re.IGNORECASE).astype(float)
    df["refund_count"]        = raw.str.count(r'\b(?:refund|reimburse|reimbursement|compensate|compensation|credit back|pay back|pay me back|money back)\b', flags=re.IGNORECASE).astype(float)
    df["regulatory_count"]    = raw.str.count(r'\b(?:CFPB|FTC|BBB|attorney general|consumer protection|OCC|FDIC|federal reserve)\b', flags=re.IGNORECASE).astype(float)
    df["urgency_count"]       = raw.str.count(r'\b(?:immediately|urgent|urgently|asap|right away|as soon as possible|promptly|right now)\b', flags=re.IGNORECASE).astype(float)
    df["paragraph_count"]     = (raw.str.count(r'\n\s*\n') + 1).astype(float)
    df["police_fraud_count"]  = raw.str.count(r'\b(?:police|filed a report|identity theft|fraud|fraudulent|stolen|theft)\b', flags=re.IGNORECASE).astype(float)
    df["quoted_speech_count"] = raw.str.count(r'"[^"]{3,}"').astype(float)
    df["question_mark_count"] = raw.str.count(r'\?').astype(float)
    df["negative_emotion_count"] = raw.str.count(r'\b(?:frustrated|frustrating|frustration|angry|anger|upset|disappointed|disappointment|shocked|outraged|furious|disgusted|distressed)\b', flags=re.IGNORECASE).astype(float)
    df["legal_count"]         = raw.str.count(r'\b(?:lawyer|attorney|sue|suing|lawsuit|court|legal action|arbitration)\b', flags=re.IGNORECASE).astype(float)

    df[TEXT_FEATURE] = raw.apply(clean_text)
    df["word_count"]      = df[TEXT_FEATURE].str.split().str.len().astype(float)
    df["avg_sentence_len"] = (df["word_count"] / df["sentence_count"].replace(0, np.nan)).fillna(0.0)

    # ── topic / categorical columns (present in dual-stratified dataset) ───────
    for col in ["issue_primary_topic", "company_primary_topic", "primary_topic"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)
    for col in df.columns:
        if col.endswith("_prob") or col in ("issue_confidence", "company_confidence", "topic_confidence"):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    for col in ["Product", "Sub-product", "Issue", "Sub-issue", "State", "Tags", "Company"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)

    return df


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(model_path: str):
    """Returns (pipeline, threshold). Handles both dict and bare pipeline artifacts."""
    artifact = joblib.load(model_path)
    if isinstance(artifact, dict):
        return artifact["pipeline"], artifact["threshold"]
    return artifact, 0.5


# ── feature importance ────────────────────────────────────────────────────────

def extract_feature_importance(pipeline, top_n: int = 200) -> pd.DataFrame:
    """
    Works for XGBoost (feature_importances_) and calibrated SVM (coef_).
    Returns only the top_n features with nonzero importance.
    Handles pipelines with an optional SelectFromModel selector step.
    """
    steps = pipeline.named_steps

    if "preprocessor" in steps:
        all_names = steps["preprocessor"].get_feature_names_out()
    elif "tfidf" in steps:
        all_names = steps["tfidf"].get_feature_names_out()
    else:
        return pd.DataFrame(columns=["feature", "importance", "rank"])

    # If a selector step exists, restrict names to selected features
    if "selector" in steps:
        mask = steps["selector"].get_support()
        feature_names = all_names[mask]
    else:
        feature_names = all_names

    # XGBoost
    if "xgb" in steps:
        importances = steps["xgb"].feature_importances_
        df = pd.DataFrame({"feature": feature_names, "importance": importances})

    # Calibrated SVM
    elif "svm" in steps or "model" in steps:
        svm_step = steps.get("svm") or steps.get("model")
        try:
            coef = svm_step.calibrated_classifiers_[0].estimator.coef_[0]
            df = pd.DataFrame({"feature": feature_names, "importance": coef})
        except AttributeError:
            return pd.DataFrame(columns=["feature", "importance", "rank"])
    else:
        return pd.DataFrame(columns=["feature", "importance", "rank"])

    return (
        df[df["importance"] > 0]
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
        .assign(rank=lambda d: range(1, len(d) + 1))
    )


# ── metrics helpers ───────────────────────────────────────────────────────────

def metrics_at_threshold(y_true, y_prob, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {"tn": cm[0][0], "fp": cm[0][1],
                             "fn": cm[1][0], "tp": cm[1][1]},
    }


# ── output saving ─────────────────────────────────────────────────────────────

def save_outputs(
    output_dir: str,
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_prob: np.ndarray,
    chosen_threshold: float,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    y_pred_default = (y_prob >= 0.5).astype(int)
    y_pred_tuned   = (y_prob >= chosen_threshold).astype(int)

    # ── 1. test_predictions.csv ───────────────────────────────────────────────
    meta_available = [c for c in META_COLS if c in X_test.columns]
    preds = X_test[meta_available].copy()
    preds["true_label"]       = y_test.values
    preds["relief_prob"]      = y_prob
    preds["pred_default"]     = y_pred_default   # threshold = 0.50
    preds["pred_tuned"]       = y_pred_tuned     # threshold = chosen_threshold
    preds["correct_tuned"]    = (y_pred_tuned == y_test.values).astype(int)
    preds["false_positive"]   = ((y_pred_tuned == 1) & (y_test.values == 0)).astype(int)
    preds["false_negative"]   = ((y_pred_tuned == 0) & (y_test.values == 1)).astype(int)
    preds.to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)
    print(f"  Saved test_predictions.csv  ({len(preds):,} rows)")

    # ── 2. pr_curve.csv ───────────────────────────────────────────────────────
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    pr_df = pd.DataFrame({
        "precision": precisions[:-1],
        "recall":    recalls[:-1],
        "threshold": thresholds,
    })
    pr_df.to_csv(os.path.join(output_dir, "pr_curve.csv"), index=False)
    print(f"  Saved pr_curve.csv          ({len(pr_df):,} points)")

    # ── 3. roc_curve.csv ──────────────────────────────────────────────────────
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thresholds})
    roc_df.to_csv(os.path.join(output_dir, "roc_curve.csv"), index=False)
    print(f"  Saved roc_curve.csv         ({len(roc_df):,} points)")

    # ── 4. feature_importance.csv ─────────────────────────────────────────────
    feat_df = extract_feature_importance(pipeline, top_n=200)
    if not feat_df.empty:
        feat_df.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)
        print(f"  Saved feature_importance.csv (top {len(feat_df):,} features with nonzero importance)")
    else:
        print(f"  Skipped feature_importance.csv (unsupported model type)")

    # ── 5. metrics.json ───────────────────────────────────────────────────────
    n_relief = int(y_test.sum())
    threshold_table = []
    for tr in [0.50, 0.70, 0.80, 0.85, 0.90, 0.95]:
        idx = np.where(recalls[:-1] >= tr)[0]
        if len(idx):
            i = idx[-1]
            threshold_table.append({
                "target_recall":  tr,
                "threshold":      float(thresholds[i]),
                "precision":      float(precisions[i]),
                "relief_caught":  int(recalls[i] * n_relief),
                "total_relief":   n_relief,
            })

    metrics = {
        "auc_roc":            float(roc_auc_score(y_test, y_prob)),
        "avg_precision":      float(average_precision_score(y_test, y_prob)),
        "chosen_threshold":   float(chosen_threshold),
        "n_test":             int(len(y_test)),
        "n_relief_test":      n_relief,
        "positive_rate":      float(y_test.mean()),
        "metrics_at_default": metrics_at_threshold(y_test, y_prob, 0.5),
        "metrics_at_chosen":  metrics_at_threshold(y_test, y_prob, chosen_threshold),
        "threshold_table":    threshold_table,
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics.json")


# ── entry point ───────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    print(f"\nLoading model: {args.model_path}")
    pipeline, threshold = load_model(args.model_path)
    if args.threshold is not None:
        threshold = args.threshold
        print(f"  Overriding threshold → {threshold:.4f}")
    else:
        print(f"  Using saved threshold  : {threshold:.4f}")

    df = load_and_prepare(args.input_csv)

    n_total  = len(df)
    n_relief = int(df["label"].sum())
    print(f"\nLabel split — relief: {n_relief:,} ({n_relief/n_total:.3%})  "
          f"no relief: {n_total - n_relief:,} ({(n_total - n_relief)/n_total:.3%})")

    df_sorted = df.sort_values("date", na_position="first").reset_index(drop=True)
    n      = len(df_sorted)
    n_test = int(n * args.test_size)
    cutoff = df_sorted["date"].iloc[n - n_test]

    X_test = df_sorted.drop(columns=["label", "date"]).iloc[n - n_test:]
    y_test = df_sorted["label"].iloc[n - n_test:]
    print(f"Chronological test set: {len(X_test):,} rows  "
          f"(from {cutoff.date()} onwards)  "
          f"relief: {int(y_test.sum()):,} / no relief: {int((y_test==0).sum()):,}")

    print("\nRunning predictions ...")
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print(f"\nSaving outputs to: {args.output_dir}")
    save_outputs(args.output_dir, pipeline, X_test, y_test, y_prob, threshold)

    print(f"\nAUC-ROC       : {roc_auc_score(y_test, y_prob):.4f}")
    print(f"Avg Precision : {average_precision_score(y_test, y_prob):.4f}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load a saved model and write prediction + analysis outputs."
    )
    parser.add_argument("--model_path",   default=DEFAULT_MODEL,
                        help="Path to saved .joblib model artifact")
    parser.add_argument("--input_csv",    default=DEFAULT_INPUT,
                        help="Input CSV (same format used at training time)")
    parser.add_argument("--output_dir",   default=DEFAULT_OUTPUT,
                        help="Directory to write all output files")
    parser.add_argument("--test_size",    type=float, default=0.2,
                        help="Fraction of most-recent rows used as test set (default 0.2)")
    parser.add_argument("--threshold",    type=float, default=None,
                        help="Override the saved threshold (default: use saved value)")
    main(parser.parse_args())
