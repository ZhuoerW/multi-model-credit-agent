"""
XGBoost — Structured Feature Model (dual-stratified dataset)
==============================================================
Feature set:

  Topic categorical : issue_primary_topic, company_primary_topic  (OHE)
  Topic numeric     : issue_confidence, company_confidence,
                      issue_topic_0..9_prob, company_topic_0..9_prob  (passthrough)
  Narrative numeric : word_count, sentence_count,
                      dollar_amount_count, max_dollar_amount  (passthrough)
  Geographic        : State  (OHE)
  Text signals      : top-20 terms from feature_importance.csv  (CountVectorizer fixed vocab)

Label: "Closed with monetary relief"
"""

import argparse
import json
import os
import re
import time
import joblib
import numpy as np
import pandas as pd
from contextlib import contextmanager

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


@contextmanager
def timer(label: str):
    print(f"[>] {label} ...", flush=True)
    t0 = time.time()
    yield
    print(f"[✓] {label} done  ({time.time() - t0:.1f}s)", flush=True)


# ── feature selector ──────────────────────────────────────────────────────────

class IndexMaskSelector(BaseEstimator, TransformerMixin):
    """Filters columns by a pre-computed boolean mask. Drop-in for SelectFromModel."""

    def __init__(self, mask: np.ndarray):
        self.mask = mask

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        import scipy.sparse as sp
        col_idx = np.where(self.mask)[0]
        return X[:, col_idx] if sp.issparse(X) else X[:, self.mask]

    def get_support(self):
        return self.mask


# ── paths ─────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT        = os.path.join(_HERE, "..", "data", "raw",
                                    "complaints_with_topics_dual_stratified.csv")
DEFAULT_FEAT_IMP     = os.path.join(_HERE, "..", "outputs", "results",
                                    "xgboost_text_only", "feature_importance.csv")
DEFAULT_OUTPUT_MODEL = os.path.join(_HERE, "..", "outputs", "models",
                                    "xgboost_structured_features.joblib")
DEFAULT_OUTPUT_DIR   = os.path.join(_HERE, "..", "outputs", "results", "xgboost_structured")
DEFAULT_SELECTED_FEATURES = os.path.join(_HERE, "..", "outputs", "results",
                                         "xgboost_structured", "feature_importance.csv")

# ── columns ───────────────────────────────────────────────────────────────────
TEXT_COL     = "Consumer complaint narrative"
RESPONSE_COL = "Company response to consumer"
TEXT_FEATURE = "clean_text"
RELIEF_RESPONSE = "Closed with monetary relief"

# topic columns present in the dual-stratified dataset
TOPIC_CAT_FEATURES = ["issue_primary_topic", "company_primary_topic"]
TOPIC_NUM_FEATURES = (
    ["issue_confidence", "company_confidence"]
    + [f"issue_topic_{i}_prob"   for i in range(10)]
    + [f"company_topic_{i}_prob" for i in range(10)]
)
GEO_CAT_FEATURES = ["State", "Company"]
NARRATIVE_NUM_FEATURES = [
     # from model_v2.ipynb
    "word_count", "sentence_count", "dollar_amount_count",
    "max_dollar_amount", "flesch_kincaid",
    "total_dollar_amount", "lexical_diversity", "char_len", "avg_sentence_len",
    "date_count", "caps_ratio", "avg_word_len", "exclamation_count",
    "account_ref_count", "duration_mention_count", "contact_attempt_count",
    "escalation_person_count", "refund_count", "regulatory_count",
    "urgency_count", "paragraph_count", "police_fraud_count",
    "quoted_speech_count", "question_mark_count", "negative_emotion_count",
    "legal_count",
]

# ── text cleaning ─────────────────────────────────────────────────────────────
_RE_REDACTION  = re.compile(r"\bx{2,}\b", re.IGNORECASE)
_RE_DOLLAR_ANY = re.compile(r"\{\$[\d,]+\.?\d*\}|\$[\d,]+\.?\d*")  # any $ pattern
_RE_PUNCT      = re.compile(r"[^a-z0-9\s]")
_RE_WHITESPACE = re.compile(r"\s+")
_RE_SENTENCE   = re.compile(r"[.!?]+")

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


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = _RE_BOILERPLATE.sub(" ", text)
    text = _RE_REDACTION.sub(" ", text)
    text = _RE_DOLLAR_ANY.sub(" dollar_amount ", text)
    text = _RE_PUNCT.sub(" ", text)
    return _RE_WHITESPACE.sub(" ", text).strip()


# ── narrative numeric features ────────────────────────────────────────────────

_RE_VOWEL_GROUP = re.compile(r'[aeiouy]+')


def count_syllables(word: str) -> int:
    word = word.lower().strip(".,!?;:'\"()-")
    if not word:
        return 0
    count = len(_RE_VOWEL_GROUP.findall(word))
    if word.endswith('e') and count > 1:
        count -= 1
    return max(1, count)


def flesch_reading_ease(text: str) -> float:
    words = str(text).split()
    if not words:
        return np.nan
    sentences = max(1, len(re.findall(r'[.!?]+', text)))
    syllables  = sum(count_syllables(w) for w in words)
    return 206.835 - 1.015 * (len(words) / sentences) - 84.6 * (syllables / len(words))


def extract_dollar_amounts(text):
    amounts = []
    for m in re.findall(r'\$[\d,]+(?:\.\d+)?', str(text)):
        try:
            amounts.append(float(m.replace('$', '').replace(',', '')))
        except ValueError:
            pass
    return amounts


def dollar_amount_count(text: str) -> int:
    return len(extract_dollar_amounts(text))


def max_dollar_amount(text: str) -> float:
    amounts = extract_dollar_amounts(text)
    return max(amounts) if amounts else 0.0


def sentence_count(text: str) -> int:
    parts = [s.strip() for s in _RE_SENTENCE.split(str(text)) if s.strip()]
    return max(len(parts), 1)


def total_dollar_amount(text: str) -> float:
    amounts = extract_dollar_amounts(text)
    return float(sum(amounts)) if amounts else 0.0


def lexical_diversity(text: str) -> float:
    words = str(text).split()
    return len(set(w.lower() for w in words)) / len(words) if words else 0.0


def caps_ratio(text: str) -> float:
    words = str(text).split()
    return sum(1 for w in words if w.isupper() and len(w) > 1) / len(words) if words else 0.0


def avg_word_len(text: str) -> float:
    words = str(text).split()
    return float(np.mean([len(w) for w in words])) if words else 0.0


# ── top-N text features from saved feature importance ────────────────────────

def load_top_text_terms(feat_imp_path: str, top_n: int = 20) -> list:
    df = pd.read_csv(feat_imp_path)
    mask = df["feature"].str.startswith("text__") | df["feature"].str.startswith("text_signals__")
    text_rows = df[mask].head(top_n)
    terms = text_rows["feature"].str.replace("^text(?:__|_signals__)", "", regex=True).tolist()
    print(f"  Top {len(terms)} text terms from {os.path.basename(feat_imp_path)}:")
    for i, t in enumerate(terms, 1):
        print(f"    {i:>2}. {t}")
    return terms


# ── data loading + preparation ────────────────────────────────────────────────

def load_and_prepare(csv_path: str) -> pd.DataFrame:
    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  Raw rows               : {len(df):,}")

    df = df.dropna(subset=[TEXT_COL])
    df = df[df[TEXT_COL].str.len() >= 50]
    df = df.dropna(subset=[RESPONSE_COL])
    df = df.drop_duplicates(subset=[TEXT_COL])
    print(f"  After deduplication    : {len(df):,}")

    df["label"] = (df[RESPONSE_COL] == RELIEF_RESPONSE).astype(int)

    # Narrative numeric features (computed on original text)
    raw = df[TEXT_COL]
    df["dollar_amount_count"] = raw.apply(dollar_amount_count).astype(float)
    df["max_dollar_amount"]   = raw.apply(max_dollar_amount).astype(float)
    df["sentence_count"]      = raw.apply(sentence_count).astype(float)
    df["flesch_kincaid"]      = raw.apply(flesch_reading_ease).astype(float)

    # v2 features (from model_v2.ipynb)
    df["total_dollar_amount"]     = raw.apply(total_dollar_amount).astype(float)
    df["lexical_diversity"]       = raw.apply(lexical_diversity).astype(float)
    df["char_len"]                = raw.str.len().astype(float)
    df["date_count"]              = raw.str.count(r'\b(?:XX|\d{1,2})/(?:XX|\d{1,2})/(?:XX|\d{2,4})\b').astype(float)
    df["caps_ratio"]              = raw.apply(caps_ratio).astype(float)
    df["avg_word_len"]            = raw.apply(avg_word_len).astype(float)
    df["exclamation_count"]       = raw.str.count(r'!').astype(float)
    df["account_ref_count"]       = raw.str.count(r'\bX{4,}\b|\b\d{4,}\b').astype(float)
    df["duration_mention_count"]  = raw.str.count(r'\b(?:month|months|week|weeks|year|years|day|days)\b', flags=re.IGNORECASE).astype(float)
    df["contact_attempt_count"]   = raw.str.count(r'\b(?:called|emailed|email|spoke with|spoken with|visited|wrote|contacted|reached out|called back|speaking with)\b', flags=re.IGNORECASE).astype(float)
    df["escalation_person_count"] = raw.str.count(r'\b(?:representative|manager|supervisor|agent|customer service|customer care|specialist|officer)\b', flags=re.IGNORECASE).astype(float)
    df["refund_count"]            = raw.str.count(r'\b(?:refund|reimburse|reimbursement|compensate|compensation|credit back|pay back|pay me back|money back)\b', flags=re.IGNORECASE).astype(float)
    df["regulatory_count"]        = raw.str.count(r'\b(?:CFPB|FTC|BBB|attorney general|consumer protection|OCC|FDIC|federal reserve)\b', flags=re.IGNORECASE).astype(float)
    df["urgency_count"]           = raw.str.count(r'\b(?:immediately|urgent|urgently|asap|right away|as soon as possible|promptly|right now)\b', flags=re.IGNORECASE).astype(float)
    df["paragraph_count"]         = (raw.str.count(r'\n\s*\n') + 1).astype(float)
    df["police_fraud_count"]      = raw.str.count(r'\b(?:police|filed a report|identity theft|fraud|fraudulent|stolen|theft)\b', flags=re.IGNORECASE).astype(float)
    df["quoted_speech_count"]     = raw.str.count(r'"[^"]{3,}"').astype(float)
    df["question_mark_count"]     = raw.str.count(r'\?').astype(float)
    df["negative_emotion_count"]  = raw.str.count(r'\b(?:frustrated|frustrating|frustration|angry|anger|upset|disappointed|disappointment|shocked|outraged|furious|disgusted|distressed)\b', flags=re.IGNORECASE).astype(float)
    df["legal_count"]             = raw.str.count(r'\b(?:lawyer|attorney|sue|suing|lawsuit|court|legal action|arbitration)\b', flags=re.IGNORECASE).astype(float)

    # Cleaned text + word count
    df[TEXT_FEATURE] = df[TEXT_COL].apply(clean_text)
    df["word_count"] = df[TEXT_FEATURE].str.split().str.len().astype(float)
    df["avg_sentence_len"] = (df["word_count"] / df["sentence_count"].replace(0, np.nan)).fillna(0.0)

    # Topic categoricals
    for col in TOPIC_CAT_FEATURES:
        df[col] = df[col].fillna("Unknown").astype(str)

    # Topic numerics
    for col in TOPIC_NUM_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Geographic
    for col in GEO_CAT_FEATURES:
        df[col] = df[col].fillna("Unknown").astype(str)

    df["date"] = pd.to_datetime(df["Date received"], errors="coerce")

    all_features = (
        TOPIC_CAT_FEATURES + TOPIC_NUM_FEATURES + GEO_CAT_FEATURES
        + NARRATIVE_NUM_FEATURES + [TEXT_FEATURE]
    )
    return df[all_features + ["label", "date"]]


# ── pipeline ──────────────────────────────────────────────────────────────────

def build_pipeline(
    topic_cat_features: list,
    geo_cat_features: list,
    num_features: list,
    top_text_terms: list,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    scale_pos_weight: float,
    random_state: int,
    prefit_selector=None,
) -> Pipeline:
    transformers = []
    if topic_cat_features:
        transformers.append((
            "topic_cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            topic_cat_features,
        ))
    if geo_cat_features:
        transformers.append((
            "geo_cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            geo_cat_features,
        ))
    if num_features:
        transformers.append(("num", "passthrough", num_features))
    if top_text_terms:
        transformers.append((
            "text_signals",
            CountVectorizer(
                vocabulary={term: i for i, term in enumerate(top_text_terms)},
                ngram_range=(1, 2),
                binary=False,
            ),
            TEXT_FEATURE,
        ))

    xgb = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        eval_metric="aucpr",
        use_label_encoder=False,
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    steps = [("preprocessor", preprocessor)]
    if prefit_selector is not None:
        steps.append(("selector", prefit_selector))
    steps.append(("xgb", xgb))
    return Pipeline(steps)


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate_pipeline(pipeline, X_val, y_val, X_test, y_test, target_recall):
    y_prob_val  = pipeline.predict_proba(X_val)[:, 1]
    y_prob_test = pipeline.predict_proba(X_test)[:, 1]

    prec_val, rec_val, thr_val = precision_recall_curve(y_val, y_prob_val)
    idx = np.where(rec_val[:-1] >= target_recall)[0]
    threshold = float(thr_val[idx[-1]]) if len(idx) else 0.5

    y_pred_default = (y_prob_test >= 0.5).astype(int)
    y_pred_tuned   = (y_prob_test >= threshold).astype(int)

    print(f"\n── Default threshold (0.50) ──")
    print(classification_report(y_test, y_pred_default,
                                target_names=["no relief", "relief"], digits=4))

    print(f"── Recall-maximising threshold ({threshold:.4f}, target recall ≥ {target_recall:.0%}) ──")
    print(classification_report(y_test, y_pred_tuned,
                                target_names=["no relief", "relief"], digits=4))

    auc  = float(roc_auc_score(y_test, y_prob_test))
    ap   = float(average_precision_score(y_test, y_prob_test))
    prec = float(precision_score(y_test, y_pred_tuned, zero_division=0))
    rec  = float(recall_score(y_test, y_pred_tuned, zero_division=0))
    f1   = float(f1_score(y_test, y_pred_tuned, zero_division=0))
    print(f"Test  AUC-ROC: {auc:.4f}   Avg Precision: {ap:.4f}   "
          f"F1: {f1:.4f}   Relief caught: {int(rec * y_test.sum())}/{int(y_test.sum())}")

    print(f"\nConfusion matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(y_test, y_pred_tuned)
    print(f"               no relief  relief")
    print(f"  no relief  {cm[0,0]:>10,} {cm[0,1]:>7,}")
    print(f"  relief     {cm[1,0]:>10,} {cm[1,1]:>7,}")

    return {
        "threshold": threshold,
        "auc_roc":       auc,
        "avg_precision": ap,
        "precision":     prec,
        "recall":        rec,
        "f1":            f1,
        "y_prob":        y_prob_test,
        "y_pred":        y_pred_tuned,
    }


def print_feature_summary(pipeline, top_n: int = 20):
    all_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    n_total   = len(all_names)

    if "selector" in pipeline.named_steps:
        mask     = pipeline.named_steps["selector"].get_support()
        selected = all_names[mask]
    else:
        selected = all_names

    importances = pipeline.named_steps["xgb"].feature_importances_
    n_nonzero   = int((importances > 0).sum())

    df = (
        pd.DataFrame({"feature": selected, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
        .assign(rank=lambda d: range(1, len(d) + 1))
    )
    print(f"\nFeature summary: {n_total} total  |  {n_nonzero} with nonzero importance")
    print(f"\nTop {top_n} features by importance:")
    for _, row in df.head(top_n).iterrows():
        print(f"  {row['feature']:<55} {row['importance']:.6f}")
    return df


def save_outputs(output_dir, pipeline, X_test, y_test, results):
    os.makedirs(output_dir, exist_ok=True)

    keep_cols = TOPIC_CAT_FEATURES + GEO_CAT_FEATURES + NARRATIVE_NUM_FEATURES
    pred_df = X_test[[c for c in keep_cols if c in X_test.columns]].copy()
    pred_df["true_label"]   = y_test.values
    pred_df["relief_prob"]  = results["y_prob"]
    pred_df["pred_tuned"]   = results["y_pred"]
    pred_df["pred_default"] = (results["y_prob"] >= 0.5).astype(int)
    pred_df.to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)
    print(f"  Saved test_predictions.csv  ({len(pred_df):,} rows)")

    feat_df = print_feature_summary(pipeline)
    feat_df.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)
    print(f"  Saved feature_importance.csv ({len(feat_df):,} features)")

    prec_arr, rec_arr, thr_arr = precision_recall_curve(y_test, results["y_prob"])
    pd.DataFrame({"precision": prec_arr[:-1], "recall": rec_arr[:-1],
                  "threshold": thr_arr}).to_csv(
        os.path.join(output_dir, "pr_curve.csv"), index=False)
    print(f"  Saved pr_curve.csv")

    metrics = {
        "auc_roc":       results["auc_roc"],
        "avg_precision": results["avg_precision"],
        "threshold":     results["threshold"],
        "precision":     results["precision"],
        "recall":        results["recall"],
        "f1":            results["f1"],
        "n_test":        int(len(y_test)),
        "n_relief_test": int(y_test.sum()),
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics.json")


# ── entry point ───────────────────────────────────────────────────────────────

def main(args):
    t_start = time.time()

    print("\nLoading top text terms ...")
    top_text_terms = load_top_text_terms(args.feat_imp_csv, top_n=args.top_text_n)

    with timer("Loading and preparing data"):
        df = load_and_prepare(args.input_csv)

    n_total  = len(df)
    n_relief = int(df["label"].sum())
    print(f"\nLabel split — relief: {n_relief:,} ({n_relief/n_total:.3%})  "
          f"no relief: {n_total - n_relief:,} ({(n_total-n_relief)/n_total:.3%})")

    feature_cols = [c for c in df.columns if c not in ("label", "date")]

    with timer("Splitting train / val / test (chronological)"):
        df_sorted = df.sort_values("date", na_position="first").reset_index(drop=True)
        n      = len(df_sorted)
        n_test = int(n * 0.2)
        n_val  = int(n * 0.2)
        n_tr   = n - n_test - n_val

        X_train = df_sorted[feature_cols].iloc[:n_tr]
        y_train = df_sorted["label"].iloc[:n_tr]
        X_val   = df_sorted[feature_cols].iloc[n_tr:n_tr + n_val]
        y_val   = df_sorted["label"].iloc[n_tr:n_tr + n_val]
        X_test  = df_sorted[feature_cols].iloc[n_tr + n_val:]
        y_test  = df_sorted["label"].iloc[n_tr + n_val:]
    print(f"  Train: {len(X_train):,}  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}")

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = (n_neg / max(n_pos, 1)) * args.spw_multiplier
    print(f"  scale_pos_weight: {n_neg/n_pos:.2f} × {args.spw_multiplier} = {scale_pos_weight:.2f}")

    active_topic_cat = TOPIC_CAT_FEATURES    if args.use_topic_cat    else []
    active_topic_num = TOPIC_NUM_FEATURES    if args.use_topic_num    else []
    active_geo       = GEO_CAT_FEATURES      if args.use_geo          else []
    active_narrative = NARRATIVE_NUM_FEATURES if args.use_narrative   else []
    active_text      = top_text_terms         if args.use_text_signals else []

    print(f"\nFeature groups (active):")
    if active_topic_cat:
        print(f"  Topic categorical ({len(active_topic_cat)})  : {active_topic_cat}  →  OHE")
    if active_topic_num:
        print(f"  Topic numeric     ({len(active_topic_num)})  : issue/company confidence + topic probs  →  passthrough")
    if active_geo:
        print(f"  Geographic        ({len(active_geo)})   : {active_geo}  →  OHE")
    if active_narrative:
        print(f"  Narrative numeric ({len(active_narrative)})  →  passthrough")
    if active_text:
        print(f"  Text signals      ({len(active_text)})  : top-{len(active_text)} terms  →  CountVectorizer")

    pipe_kwargs = dict(
        topic_cat_features=active_topic_cat,
        geo_cat_features=active_geo,
        num_features=active_topic_num + active_narrative,
        top_text_terms=active_text,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        scale_pos_weight=scale_pos_weight,
        random_state=args.random_state,
    )

    prefit_selector = None
    sel_csv = args.selected_features_csv
    if sel_csv and os.path.exists(sel_csv):
        selected_names = set(pd.read_csv(sel_csv)["feature"].tolist())
        temp_pp = build_pipeline(**pipe_kwargs).named_steps["preprocessor"]
        temp_pp.fit(X_train)
        all_names = temp_pp.get_feature_names_out()
        mask = np.array([n in selected_names for n in all_names], dtype=bool)
        n_kept = int(mask.sum())
        if n_kept > 0:
            print(f"  Loaded {n_kept}/{len(all_names)} pre-selected features "
                  f"from {os.path.basename(sel_csv)} — skipping Pass 1")
            prefit_selector = IndexMaskSelector(mask)

    with timer("Training XGBoost"):
        if prefit_selector is not None:
            pipeline = build_pipeline(**pipe_kwargs, prefit_selector=prefit_selector)
            pipeline.fit(X_train, y_train)
        else:
            # Pass 1: train on all features to identify zero-importance ones
            pipe1 = build_pipeline(**pipe_kwargs)
            pipe1.fit(X_train, y_train)

            xgb1_imp = pipe1.named_steps["xgb"].feature_importances_
            n_all    = len(xgb1_imp)
            selector = SelectFromModel(pipe1.named_steps["xgb"], prefit=True, threshold=1e-10)
            n_kept   = int(selector.get_support().sum())
            print(f"  Pass 1: {n_all} features  →  {n_kept} nonzero-importance kept")

            if n_kept < n_all:
                # Pass 2: retrain on the reduced feature set
                pipe2 = build_pipeline(**pipe_kwargs, prefit_selector=selector)
                pipe2.fit(X_train, y_train)
                pipeline = pipe2
            else:
                pipeline = pipe1

    with timer("Evaluating"):
        results = evaluate_pipeline(pipeline, X_val, y_val, X_test, y_test, args.target_recall)

    print(f"\nSaving outputs to: {args.output_dir}")
    save_outputs(args.output_dir, pipeline, X_test, y_test, results)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_model)), exist_ok=True)
    joblib.dump({"pipeline": pipeline, "threshold": results["threshold"],
                 "top_text_terms": top_text_terms}, args.output_model)
    print(f"Saved model → {args.output_model}")
    print(f"\nTotal time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="XGBoost with structured features on dual-stratified dataset."
    )
    parser.add_argument("--input_csv",              default=DEFAULT_INPUT)
    parser.add_argument("--feat_imp_csv",           default=DEFAULT_FEAT_IMP)
    parser.add_argument("--selected_features_csv",  default=DEFAULT_SELECTED_FEATURES,
                        help="CSV of pre-selected features from a prior run; "
                             "if the file exists Pass 1 is skipped.")
    parser.add_argument("--output_model",   default=DEFAULT_OUTPUT_MODEL)
    parser.add_argument("--output_dir",     default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top_text_n",     type=int,   default=50)
    # feature group toggles
    parser.add_argument("--use_topic_cat",    action=argparse.BooleanOptionalAction, default=True,
                        help="Include topic categorical features (OHE)")
    parser.add_argument("--use_topic_num",    action=argparse.BooleanOptionalAction, default=False,
                        help="Include topic numeric features (confidence + prob vectors)")
    parser.add_argument("--use_geo",          action=argparse.BooleanOptionalAction, default=True,
                        help="Include geographic features (State, Company OHE)")
    parser.add_argument("--use_narrative",    action=argparse.BooleanOptionalAction, default=True,
                        help="Include 26 narrative numeric features")
    parser.add_argument("--use_text_signals", action=argparse.BooleanOptionalAction, default=True,
                        help="Include top-N text term signals (CountVectorizer)")
    parser.add_argument("--random_state",   type=int,   default=42)
    parser.add_argument("--n_estimators",   type=int,   default=500)
    parser.add_argument("--max_depth",      type=int,   default=6)
    parser.add_argument("--learning_rate",  type=float, default=0.05)
    parser.add_argument("--spw_multiplier", type=float, default=1.5)
    parser.add_argument("--target_recall",  type=float, default=0.85)
    main(parser.parse_args())
