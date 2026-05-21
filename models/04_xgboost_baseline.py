"""
XGBoost Relief Classifier — CFPB Consumer Complaints 
=======================================================================
Use case:
    A consumer submits a complaint to a company.  Given only the complaint
    narrative, predict whether the company will ultimately close the case
    with monetary relief — i.e. the consumer gets paid.

    Input:
        - Consumer complaint narrative  →  TF-IDF bigrams
        - primary_topic                →  one-hot  (LDA/BERTopic topic ID)

    Label  (derived from Company response to consumer):
        relief = 1  →  "Closed with monetary relief"
        relief = 0  →  all other responses (closed without relief, in progress, etc.)

    Class imbalance:
        Monetary relief is a minority outcome (~15-20 % of closed complaints).
        scale_pos_weight compensates; use Average Precision and AUC-ROC
        as primary metrics, not accuracy.

    NOT used as features:
        - Product, Sub-product, Issue, Sub-issue, State, Tags  →  ablation experiment
        - Company response to consumer   →  this is the label
        - Company public response        →  correlated post-complaint outcome
        - Consumer disputed              →  99.85 % missing
        - ZIP code                       →  6,930 partially-redacted values, too noisy
        - Timely response, Submitted via, Consumer consent  →  zero variance
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


@contextmanager
def timer(label: str):
    print(f"[>] {label} ...", flush=True)
    t0 = time.time()
    yield
    print(f"[✓] {label} done  ({time.time() - t0:.1f}s)", flush=True)


from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


# ── paths ─────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT  = os.path.join(_HERE, "..", "data", "raw", "complaints_with_topics.csv")
DEFAULT_OUTPUT      = os.path.join(_HERE, "..", "outputs", "models", "xgboost_relief_classifier_with_topics.joblib")
DEFAULT_OUTPUT_TEXT = os.path.join(_HERE, "..", "outputs", "models", "xgboost_relief_classifier_text_only.joblib")

# ── columns ───────────────────────────────────────────────────────────────────
TEXT_COL             = "Consumer complaint narrative"
RESPONSE_COL         = "Company response to consumer"

TEXT_FEATURE     = "clean_text"
CAT_FEATURES     = ["primary_topic"]

# ── label definition ──────────────────────────────────────────────────────────
RELIEF_RESPONSE = "Closed with monetary relief"

# ── text cleaning ─────────────────────────────────────────────────────────────
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


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = _RE_BOILERPLATE.sub(" ", text)
    text = _RE_REDACTION.sub(" ", text)
    text = _RE_DIGITS.sub(" ", text)
    text = _RE_PUNCT.sub(" ", text)
    text = _RE_WHITESPACE.sub(" ", text).strip()
    return text


# ── data loading + preparation ────────────────────────────────────────────────

def load_and_prepare(csv_path: str) -> pd.DataFrame:
    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  Raw rows               : {len(df):,}")

    # Require a substantive narrative
    df = df.dropna(subset=[TEXT_COL])
    df = df[df[TEXT_COL].str.len() >= 50]
    print(f"  After narrative filter : {len(df):,}")

    # Drop rows with no company response (complaint still open / no outcome yet)
    df = df.dropna(subset=[RESPONSE_COL])
    print(f"  After response filter  : {len(df):,}")

    # Deduplicate on narrative text to prevent train/test leakage
    df = df.drop_duplicates(subset=[TEXT_COL])
    print(f"  After deduplication    : {len(df):,}")

    # Derive relief label from Company response to consumer
    df["label"] = (df[RESPONSE_COL] == RELIEF_RESPONSE).astype(int)

    # Clean narrative text
    df[TEXT_FEATURE] = df[TEXT_COL].apply(clean_text)

    # Cast primary_topic to str for OHE; fill any missing topic values
    for col in CAT_FEATURES:
        df[col] = df[col].fillna("Unknown").astype(str)
    return df[[TEXT_FEATURE] + CAT_FEATURES + ["label"]]


# ── model pipeline ────────────────────────────────────────────────────────────

def build_pipeline(
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    scale_pos_weight: float,
    random_state: int,
    use_topics: bool = True,
) -> Pipeline:
    """
    Both variants take the same DataFrame as input; ColumnTransformer selects
    only the columns it needs via remainder="drop".

    use_topics=True  → TF-IDF + OHE(primary_topic)
    use_topics=False → TF-IDF only (ablation baseline)

    All outputs are kept sparse so XGBoost hist can handle 30K+ features
    without densifying the matrix.
    """
    transformers = [
        (
            "text",
            TfidfVectorizer(
                max_features=30_000,
                ngram_range=(1, 2),
                min_df=3,
                max_df=0.95,
                stop_words="english",
                sublinear_tf=True,
            ),
            TEXT_FEATURE,
        ),
    ]
    if use_topics:
        transformers += [
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                CAT_FEATURES,
            ),
        ]

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
        verbosity=1,
    )

    return Pipeline([
        ("preprocessor", ColumnTransformer(transformers=transformers, remainder="drop")),
        ("xgb", xgb),
    ])


# ── evaluation ────────────────────────────────────────────────────────────────

def print_confusion_matrix(cm: np.ndarray) -> None:
    print("Confusion matrix  (rows = actual, cols = predicted):")
    print(f"               no relief  relief")
    print(f"  no relief  {cm[0, 0]:>10,} {cm[0, 1]:>7,}")
    print(f"  relief     {cm[1, 0]:>10,} {cm[1, 1]:>7,}")


def print_top_features(pipeline: Pipeline, n: int = 15) -> None:
    """Show features with highest XGBoost importance scores."""
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = pipeline.named_steps["xgb"].feature_importances_
    top_idx = np.argsort(importances)[-n:][::-1]
    top_features = [(feature_names[i], importances[i]) for i in top_idx]
    print(f"\nTop {n} features by XGBoost importance (gain):")
    for feat, score in top_features:
        print(f"  {feat:<40} {score:.6f}")


# ── evaluation helpers ────────────────────────────────────────────────────────

def evaluate_pipeline(
    pipeline: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_recall: float,
    label: str,
) -> dict:
    """
    Threshold is selected on the validation set, then metrics are reported
    on the held-out test set.  This prevents the threshold from being
    overfitted to the test set.
    """
    # ── val: threshold selection only ─────────────────────────────────────────
    y_prob_val = pipeline.predict_proba(X_val)[:, 1]
    prec_val, rec_val, thr_val = precision_recall_curve(y_val, y_prob_val)
    idx = np.where(rec_val[:-1] >= target_recall)[0]
    if len(idx):
        chosen_threshold = float(thr_val[idx[-1]])
    else:
        chosen_threshold = 0.5
        print(f"  [!] {label}: target recall {target_recall:.0%} not achievable on val set; "
              f"falling back to 0.5")

    # ── test: final evaluation at val-chosen threshold ────────────────────────
    y_prob_test = pipeline.predict_proba(X_test)[:, 1]
    prec_test, rec_test, thr_test = precision_recall_curve(y_test, y_prob_test)
    y_pred = (y_prob_test >= chosen_threshold).astype(int)

    return {
        "label":             label,
        "threshold":         chosen_threshold,          # chosen on VAL
        # test-set metrics (authoritative)
        "y_prob":            y_prob_test,
        "auc_roc":           float(roc_auc_score(y_test, y_prob_test)),
        "avg_precision":     float(average_precision_score(y_test, y_prob_test)),
        "precision":         float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":            float(recall_score(y_test, y_pred, zero_division=0)),
        "f1":                float(f1_score(y_test, y_pred, zero_division=0)),
        "precisions":        prec_test,
        "recalls":           rec_test,
        "thresholds":        thr_test,
        # val-set metrics (for sanity-checking val/test alignment)
        "val_auc_roc":       float(roc_auc_score(y_val, y_prob_val)),
        "val_avg_precision": float(average_precision_score(y_val, y_prob_val)),
        "val_precisions":    prec_val,
        "val_recalls":       rec_val,
        "val_thresholds":    thr_val,
    }


def print_comparison(r_text: dict, r_topics: dict, n_relief_test: int) -> None:
    """Side-by-side metric table: text-only vs text+topics."""
    w = 56
    print("\n" + "═" * w)
    print(f"  {'Metric':<24} {'Text only':>10} {'+ Topics':>10} {'Δ':>8}")
    print("═" * w)
    rows = [
        ("AUC-ROC",                 "auc_roc"),
        ("Avg Precision",           "avg_precision"),
        (f"Precision @ target R",   "precision"),
        ("Recall (actual)",         "recall"),
        ("F1",                      "f1"),
    ]
    for name, key in rows:
        v_text   = r_text[key]
        v_topics = r_topics[key]
        delta    = v_topics - v_text
        sign     = "+" if delta >= 0 else ""
        print(f"  {name:<24} {v_text:>10.4f} {v_topics:>10.4f} {sign}{delta:>7.4f}")
    print(f"  {'Chosen threshold':<24} {r_text['threshold']:>10.4f} {r_topics['threshold']:>10.4f}")

    # Relief caught at chosen threshold
    caught_text   = int(r_text["recall"]   * n_relief_test)
    caught_topics = int(r_topics["recall"] * n_relief_test)
    print(f"  {'Relief caught':<24} {caught_text:>9}/{n_relief_test} {caught_topics:>9}/{n_relief_test}")
    print("═" * w)
    winner = "+ Topics" if r_topics["auc_roc"] >= r_text["auc_roc"] else "Text only"
    print(f"  Best model by AUC-ROC: {winner}")
    print("═" * w)


# ── entry point ───────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    t_start = time.time()

    with timer("Loading and preparing data"):
        df = load_and_prepare(args.input_csv)

    n_total = len(df)
    n_relief = int(df["label"].sum())
    n_no_relief = n_total - n_relief
    print(f"\nLabel split — relief: {n_relief:,} ({n_relief/n_total:.3%})  "
          f"no relief: {n_no_relief:,} ({n_no_relief/n_total:.3%})")

    X = df.drop(columns=["label"])
    y = df["label"]

    with timer("Splitting train / val / test"):
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=y,
        )
        val_frac = args.val_size / (1.0 - args.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=val_frac,
            random_state=args.random_state,
            stratify=y_trainval,
        )
    print(f"    Train: {len(X_train):,}  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}")

    if args.undersample_ratio > 0:
        from sklearn.utils import resample
        neg_idx = X_train.index[y_train == 0]
        pos_idx = X_train.index[y_train == 1]
        n_keep = int(len(pos_idx) * args.undersample_ratio)
        neg_down = resample(neg_idx, n_samples=n_keep, replace=False,
                            random_state=args.random_state)
        keep = neg_down.tolist() + pos_idx.tolist()
        X_train, y_train = X_train.loc[keep], y_train.loc[keep]
        print(f"  After undersampling — no relief: {(y_train==0).sum():,}  "
              f"relief: {(y_train==1).sum():,}  ratio: {args.undersample_ratio:.0f}:1")

    # Compute scale_pos_weight from training set unless overridden
    if args.scale_pos_weight <= 0:
        n_train_neg = int((y_train == 0).sum())
        n_train_pos = int((y_train == 1).sum())
        natural_spw = n_train_neg / max(n_train_pos, 1)
        scale_pos_weight = natural_spw * args.spw_multiplier
        print(f"  Auto scale_pos_weight  : {natural_spw:.2f} × {args.spw_multiplier} "
              f"= {scale_pos_weight:.2f}  "
              f"({n_train_neg:,} no-relief / {n_train_pos:,} relief)")
    else:
        scale_pos_weight = args.scale_pos_weight

    pipeline_kw = dict(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        scale_pos_weight=scale_pos_weight,
        random_state=args.random_state,
    )

    # ── variant A: text only ──────────────────────────────────────────────────
    with timer("Training  [A] Text only  (TF-IDF → XGBoost)"):
        pipe_text = build_pipeline(**pipeline_kw, use_topics=False)
        pipe_text.fit(X_train, y_train)

    # ── variant B: text + topics ──────────────────────────────────────────────
    with timer("Training  [B] Text + topics  (TF-IDF + topic OHE/num → XGBoost)"):
        pipe_topics = build_pipeline(**pipeline_kw, use_topics=True)
        pipe_topics.fit(X_train, y_train)

    # ── evaluate both ─────────────────────────────────────────────────────────
    n_relief_test = int(y_test.sum())
    target_recall = args.target_recall

    with timer("Evaluating both variants"):
        r_text   = evaluate_pipeline(pipe_text,   X_val, y_val, X_test, y_test, target_recall, "Text only")
        r_topics = evaluate_pipeline(pipe_topics, X_val, y_val, X_test, y_test, target_recall, "Text + topics")

    # ── per-variant detailed output ───────────────────────────────────────────
    for r, pipe in [(r_text, pipe_text), (r_topics, pipe_topics)]:
        thr = r["threshold"]
        print(f"\n{'━'*60}")
        print(f"  {r['label']}  (threshold = {thr:.4f}, target recall ≥ {target_recall:.0%})")
        print(f"{'━'*60}")

        print(f"\n── Default threshold (0.50) ──")
        y_pred_default = (r["y_prob"] >= 0.5).astype(int)
        print(classification_report(y_test, y_pred_default,
                                    target_names=["no relief", "relief"], digits=4))
        print_confusion_matrix(confusion_matrix(y_test, y_pred_default))

        print(f"\n── Recall-maximising threshold ({thr:.4f}, selected on val set) ──")
        y_pred_tuned = (r["y_prob"] >= thr).astype(int)
        print(classification_report(y_test, y_pred_tuned,
                                    target_names=["no relief", "relief"], digits=4))
        print_confusion_matrix(confusion_matrix(y_test, y_pred_tuned))

        print(f"\nVal  AUC-ROC   : {r['val_auc_roc']:.4f}   Avg Precision: {r['val_avg_precision']:.4f}")
        print(f"Test AUC-ROC   : {r['auc_roc']:.4f}   Avg Precision: {r['avg_precision']:.4f}")

        print_top_features(pipe)

        print(f"\nThreshold selection (val set) — test-set relief caught at each target:")
        print(f"  {'Target recall':<16} {'Threshold':>10} {'Val Prec':>10} {'Relief caught':>15}")
        prec_v, rec_v, thr_v = r["val_precisions"], r["val_recalls"], r["val_thresholds"]
        rec_t = r["recalls"]
        for tr in [0.50, 0.70, 0.80, 0.85, 0.90]:
            idx_v = np.where(rec_v[:-1] >= tr)[0]
            if len(idx_v):
                i_v = idx_v[-1]
                sel_thr = thr_v[i_v]
                # find closest test-set operating point at this threshold
                idx_t = np.where(r["thresholds"] <= sel_thr)[0]
                caught = int(rec_t[idx_t[-1]] * n_relief_test) if len(idx_t) else 0
                marker = " ◀ chosen" if abs(tr - target_recall) < 1e-6 else ""
                print(f"  recall≥{tr:.0%}        {sel_thr:>10.4f} "
                      f"{prec_v[i_v]:>10.4f} {caught:>8}/{n_relief_test}{marker}")

    # ── side-by-side comparison ───────────────────────────────────────────────
    print_comparison(r_text, r_topics, n_relief_test)

    # ── save both models ──────────────────────────────────────────────────────
    with timer("Saving models"):
        os.makedirs(os.path.dirname(os.path.abspath(args.output_model)), exist_ok=True)
        joblib.dump({"pipeline": pipe_topics, "threshold": r_topics["threshold"]},
                    args.output_model)
        joblib.dump({"pipeline": pipe_text,   "threshold": r_text["threshold"]},
                    args.output_model_text)
    print(f"    Saved (topics)    → {args.output_model}")
    print(f"    Saved (text only) → {args.output_model_text}")

    print(f"\nTotal time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XGBoost classifier predicting monetary relief outcome.")
    parser.add_argument("--input_csv",         default=DEFAULT_INPUT)
    parser.add_argument("--output_model",       default=DEFAULT_OUTPUT,
                        help="Path for text+topics model artifact")
    parser.add_argument("--output_model_text",  default=DEFAULT_OUTPUT_TEXT,
                        help="Path for text-only model artifact")
    parser.add_argument("--test_size",          type=float, default=0.2)
    parser.add_argument("--val_size",           type=float, default=0.2,
                        help="Validation fraction of total data (default 0.2); "
                             "threshold is selected on val, final metrics on test")
    parser.add_argument("--random_state",       type=int,   default=42)
    parser.add_argument("--n_estimators",       type=int,   default=500,
                        help="Number of boosting rounds")
    parser.add_argument("--max_depth",          type=int,   default=4,
                        help="Maximum tree depth")
    parser.add_argument("--learning_rate",      type=float, default=0.05,
                        help="Boosting learning rate (eta)")
    parser.add_argument("--scale_pos_weight",   type=float, default=-1,
                        help="Class weight for positives (<=0 = auto from training set)")
    parser.add_argument("--spw_multiplier",     type=float, default=1.0,
                        help="Multiply auto scale_pos_weight by this factor to boost recall "
                             "(ignored when --scale_pos_weight is set manually)")
    parser.add_argument("--undersample_ratio",  type=float, default=0,
                        help="No-relief:relief ratio after undersampling (0 = disabled)")
    parser.add_argument("--target_recall",      type=float, default=0.85,
                        help="Minimum recall for the relief class when choosing the "
                             "decision threshold (0.0 = use default 0.5)")
    main(parser.parse_args())
