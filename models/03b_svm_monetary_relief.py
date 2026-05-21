import json
import os
import re
import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    brier_score_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC


TEXT_COL = "Consumer complaint narrative"
RESPONSE_COL = "Company response to consumer"

TEXT_FEATURE = "clean_text"
CAT_FEATURES = ["primary_topic"]

POSITIVE_RESPONSE = "Closed with monetary relief"
VALID_RESPONSES = {
    "Closed with monetary relief",
    "Closed with non-monetary relief",
    "Closed with explanation",
}

_RE_REDACTION = re.compile(r"\bx{2,}\b", re.IGNORECASE)
_RE_DIGITS = re.compile(r"\b\d+\b")
_RE_PUNCT = re.compile(r"[^a-z\s]")
_RE_WHITESPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = _RE_REDACTION.sub(" ", text)
    text = _RE_DIGITS.sub(" ", text)
    text = _RE_PUNCT.sub(" ", text)
    text = _RE_WHITESPACE.sub(" ", text).strip()
    return text


def load_and_prepare(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)

    df = df[df[RESPONSE_COL].isin(VALID_RESPONSES)]
    df = df.dropna(subset=[TEXT_COL])
    df = df[df[TEXT_COL].str.len() >= 50]
    df = df.drop_duplicates(subset=[TEXT_COL])

    df["label"] = (df[RESPONSE_COL] == POSITIVE_RESPONSE).astype(int)
    df[TEXT_FEATURE] = df[TEXT_COL].apply(clean_text)

    for col in CAT_FEATURES:
        df[col] = df[col].fillna("Unknown").astype(str)

    return df[[TEXT_FEATURE] + CAT_FEATURES + ["label"]]


def build_pipeline(C: float, random_state: int, use_topics: bool = True) -> Pipeline:
    """
    use_topics=True  → TF-IDF + OHE(primary_topic)
    use_topics=False → TF-IDF only (ablation baseline)

    Both variants accept the same DataFrame; ColumnTransformer with
    remainder="drop" ignores columns the variant doesn't need.
    """
    transformers = [
        (
            "text",
            TfidfVectorizer(
                max_features=60_000,
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

    base_svm = LinearSVC(
        C=C,
        class_weight="balanced",
        max_iter=5000,
        random_state=random_state,
    )

    return Pipeline([
        ("preprocessor", ColumnTransformer(transformers=transformers, remainder="drop")),
        ("model", CalibratedClassifierCV(estimator=base_svm, method="sigmoid", cv=3)),
    ])


def select_threshold(y_true, y_prob, strategy="recall_constraint", target_recall=0.85, beta=2.0):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    precision = precision[:-1]
    recall = recall[:-1]

    if strategy == "recall_constraint":
        idx = np.where(recall >= target_recall)[0]
        if len(idx) == 0:
            return 0.5
        best_idx = idx[np.argmax(precision[idx])]
        return float(thresholds[best_idx])

    elif strategy == "f_beta":
        beta2 = beta ** 2
        f_beta = (1 + beta2) * precision * recall / (beta2 * precision + recall + 1e-12)
        return float(thresholds[np.argmax(f_beta)])

    elif strategy == "max_f1":
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        return float(thresholds[np.argmax(f1)])

    else:
        raise ValueError(f"Unknown threshold strategy: {strategy}")


def compute_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold":          float(threshold),
        "roc_auc":            float(roc_auc_score(y_true, y_prob)),
        "average_precision":  float(average_precision_score(y_true, y_prob)),
        "precision":          float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":             float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":                 float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy":  float(balanced_accuracy_score(y_true, y_pred)),
        "mcc":                float(matthews_corrcoef(y_true, y_pred)),
        "brier_score":        float(brier_score_loss(y_true, y_prob)),
        "confusion_matrix":   confusion_matrix(y_true, y_pred).tolist(),
    }


def bootstrap_ci(y_true, y_prob, threshold, n_bootstrap=1000, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    rows = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        rows.append(compute_metrics(y_true[idx], y_prob[idx], threshold))

    df = pd.DataFrame(rows)
    ci = {}
    for col in ["roc_auc", "average_precision", "precision", "recall",
                "f1", "balanced_accuracy", "mcc", "brier_score"]:
        ci[col] = {
            "mean":     float(df[col].mean()),
            "ci_lower": float(df[col].quantile(0.025)),
            "ci_upper": float(df[col].quantile(0.975)),
        }
    return ci


def print_threshold_table(y_true, y_prob, label=""):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    header = f"Validation threshold table{' — ' + label if label else ''}:"
    print(f"\n{header}")
    print(f"{'Target recall':<16} {'Threshold':>12} {'Precision':>12}")
    for target in [0.50, 0.70, 0.80, 0.85, 0.90, 0.95]:
        idx = np.where(recall[:-1] >= target)[0]
        if len(idx) == 0:
            print(f"recall≥{target:.0%}         {'N/A':>12} {'N/A':>12}")
            continue
        best_idx = idx[np.argmax(precision[:-1][idx])]
        print(f"recall≥{target:.0%}       {thresholds[best_idx]:>12.4f} {precision[best_idx]:>12.4f}")


def evaluate_variant(
    pipeline,
    X_val, y_val,
    X_test, y_test,
    threshold_strategy, target_recall, beta,
    n_bootstrap, seed,
    label,
):
    """Threshold selected on val set; all reported metrics on test set."""
    val_prob  = pipeline.predict_proba(X_val)[:, 1]
    test_prob = pipeline.predict_proba(X_test)[:, 1]

    print_threshold_table(y_val, val_prob, label=label)

    threshold = select_threshold(
        y_val, val_prob,
        strategy=threshold_strategy,
        target_recall=target_recall,
        beta=beta,
    )
    print(f"  Selected threshold ({label}): {threshold:.4f}")

    val_metrics  = compute_metrics(y_val,  val_prob,  threshold)
    test_metrics = compute_metrics(y_test, test_prob, threshold)

    test_pred = (test_prob >= threshold).astype(int)
    print(f"\nTest classification report ({label}):")
    print(classification_report(y_test, test_pred,
                                target_names=["other", "monetary relief"], digits=4))

    print(f"Computing bootstrap CI ({label}) ...")
    test_ci = bootstrap_ci(y_test, test_prob, threshold,
                           n_bootstrap=n_bootstrap, seed=seed)

    return {
        "label":        label,
        "threshold":    threshold,
        "val_metrics":  val_metrics,
        "test_metrics": test_metrics,
        "test_ci":      test_ci,
        "test_prob":    test_prob,
        "test_pred":    test_pred,
    }


def print_comparison(r_text: dict, r_topics: dict, n_relief_test: int) -> None:
    w = 62
    print("\n" + "═" * w)
    print(f"  {'Metric':<24} {'Text only':>14} {'+ Topics':>14} {'Δ':>6}")
    print("═" * w)
    rows = [
        ("AUC-ROC",        "roc_auc"),
        ("Avg Precision",  "average_precision"),
        ("Precision",      "precision"),
        ("Recall",         "recall"),
        ("F1",             "f1"),
        ("Brier Score",    "brier_score"),
    ]
    for name, key in rows:
        v_text   = r_text["test_metrics"][key]
        v_topics = r_topics["test_metrics"][key]
        delta    = v_topics - v_text
        sign     = "+" if delta >= 0 else ""
        print(f"  {name:<24} {v_text:>14.4f} {v_topics:>14.4f} {sign}{delta:>5.4f}")
    print(f"  {'Chosen threshold':<24} {r_text['threshold']:>14.4f} {r_topics['threshold']:>14.4f}")
    caught_text   = int(r_text["test_metrics"]["recall"]   * n_relief_test)
    caught_topics = int(r_topics["test_metrics"]["recall"] * n_relief_test)
    print(f"  {'Relief caught':<24} {caught_text:>13}/{n_relief_test} {caught_topics:>13}/{n_relief_test}")
    print("═" * w)
    winner = "+ Topics" if r_topics["test_metrics"]["roc_auc"] >= r_text["test_metrics"]["roc_auc"] else "Text only"
    print(f"  Best model by AUC-ROC: {winner}")
    print("═" * w)


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    df = load_and_prepare(args.input_csv)
    X = df.drop(columns=["label"])
    y = df["label"]

    print(f"Total samples: {len(df):,}")
    print(f"Positive rate: {y.mean():.4%}")

    # 60% train / 20% val / 20% test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=args.random_state, stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=0.25,  # 0.25 × 0.80 = 0.20 of total
        random_state=args.random_state, stratify=y_trainval,
    )
    print(f"Train: {len(X_train):,}  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}")

    eval_kw = dict(
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        threshold_strategy=args.threshold_strategy,
        target_recall=args.target_recall,
        beta=args.beta,
        n_bootstrap=args.bootstrap,
        seed=args.random_state,
    )

    # ── variant A: text only ──────────────────────────────────────────────────
    print("\n" + "━" * 60)
    print("  [A] Text only  (TF-IDF → SVM)")
    print("━" * 60)
    pipe_text = build_pipeline(args.C, args.random_state, use_topics=False)
    print("Training [A] ...")
    pipe_text.fit(X_train, y_train)
    r_text = evaluate_variant(pipe_text, label="Text only", **eval_kw)

    # ── variant B: text + topics ──────────────────────────────────────────────
    print("\n" + "━" * 60)
    print("  [B] Text + topics  (TF-IDF + OHE(primary_topic) → SVM)")
    print("━" * 60)
    pipe_topics = build_pipeline(args.C, args.random_state, use_topics=True)
    print("Training [B] ...")
    pipe_topics.fit(X_train, y_train)
    r_topics = evaluate_variant(pipe_topics, label="Text + topics", **eval_kw)

    # ── comparison ────────────────────────────────────────────────────────────
    n_relief_test = int(y_test.sum())
    print_comparison(r_text, r_topics, n_relief_test)

    # ── save both variants ────────────────────────────────────────────────────
    for r, pipe, suffix in [
        (r_text,   pipe_text,   "text_only"),
        (r_topics, pipe_topics, "with_topics"),
    ]:
        results = {
            "label_positive": POSITIVE_RESPONSE,
            "class_balance": {
                "positive_rate": float(y.mean()),
                "n_total":       int(len(df)),
                "n_positive":    int(y.sum()),
                "n_negative":    int(len(df) - y.sum()),
            },
            "split_sizes": {
                "train":      int(len(X_train)),
                "validation": int(len(X_val)),
                "test":       int(len(X_test)),
            },
            "threshold_selection": {
                "strategy":           args.threshold_strategy,
                "target_recall":      args.target_recall,
                "beta":               args.beta,
                "selected_threshold": r["threshold"],
                "selected_on":        "validation set",
            },
            "validation_metrics": r["val_metrics"],
            "test_metrics":       r["test_metrics"],
            "test_bootstrap_ci":  r["test_ci"],
        }

        metrics_path = os.path.join(args.output_dir, f"svm_monetary_relief_{suffix}_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(results, f, indent=2)

        pred_df = X_test.copy()
        pred_df["y_true"] = y_test.values
        pred_df["y_prob"] = r["test_prob"]
        pred_df["y_pred"] = r["test_pred"]
        pred_path = os.path.join(args.output_dir, f"svm_monetary_relief_{suffix}_predictions.csv")
        pred_df.to_csv(pred_path, index=False)

        model_path = os.path.join(args.output_dir, f"svm_monetary_relief_{suffix}_model.joblib")
        joblib.dump(
            {
                "pipeline":           pipe,
                "threshold":          r["threshold"],
                "threshold_strategy": args.threshold_strategy,
                "target_recall":      args.target_recall,
                "metrics":            results,
            },
            model_path,
        )
        print(f"Saved {suffix}: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_csv",  required=True)
    parser.add_argument("--output_dir", default="outputs/models")
    parser.add_argument("--random_state", type=int,   default=42)
    parser.add_argument("--C",            type=float, default=1.0)

    parser.add_argument(
        "--threshold_strategy",
        choices=["recall_constraint", "f_beta", "max_f1"],
        default="recall_constraint",
    )

    parser.add_argument("--target_recall", type=float, default=0.85)
    parser.add_argument("--beta",          type=float, default=2.0)
    parser.add_argument("--bootstrap",     type=int,   default=1000)

    args = parser.parse_args()
    main(args)
