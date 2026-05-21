"""
visualize_curves.py — Generate evaluation plots from evaluate_model.py outputs.

Produces three figures saved to --output_dir:
    pr_curve.png              Precision–Recall curve
    threshold_curve.png       Threshold vs Precision / Recall
    roc_curve.png             ROC curve

Usage:
    python visualize_curves.py
    python visualize_curves.py --results_dir ../outputs/results/xgboost_with_topics
                               --output_dir  ../outputs/results/xgboost_with_topics/plots
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# ── paths ─────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS = os.path.join(_HERE, "..", "outputs", "results", "xgboost_with_topics")

# ── style ─────────────────────────────────────────────────────────────────────
BLUE   = "#2563EB"
ORANGE = "#EA580C"
GREEN  = "#16A34A"
GRAY   = "#9CA3AF"
RED    = "#DC2626"

plt.rcParams.update({
    "figure.dpi":        150,
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "grid.linestyle":    "--",
})


# ── helpers ───────────────────────────────────────────────────────────────────

def load_data(results_dir: str):
    pr   = pd.read_csv(os.path.join(results_dir, "pr_curve.csv"))
    roc  = pd.read_csv(os.path.join(results_dir, "roc_curve.csv"))
    with open(os.path.join(results_dir, "metrics.json")) as f:
        metrics = json.load(f)
    return pr, roc, metrics


def _mark_threshold(ax, x, y, label, color=RED, marker="o", ms=9):
    """Plot a marker and annotate it."""
    ax.scatter([x], [y], color=color, s=ms ** 2, zorder=5, marker=marker)
    ax.annotate(
        label,
        xy=(x, y),
        xytext=(10, -14),
        textcoords="offset points",
        fontsize=9,
        color=color,
        arrowprops=dict(arrowstyle="-", color=color, lw=0.8),
    )


# ── plot 1: Precision–Recall curve ───────────────────────────────────────────

def plot_pr_curve(pr: pd.DataFrame, metrics: dict, output_path: str) -> None:
    avg_p   = metrics["avg_precision"]
    pos_rate = metrics["positive_rate"]
    chosen  = metrics["chosen_threshold"]
    m_chosen = metrics["metrics_at_chosen"]

    fig, ax = plt.subplots(figsize=(7, 5))

    # Main curve
    ax.plot(pr["recall"], pr["precision"], color=BLUE, lw=2,
            label=f"Model  (AP = {avg_p:.3f})")

    # No-skill baseline
    ax.axhline(pos_rate, color=GRAY, lw=1.2, linestyle="--",
               label=f"No-skill baseline  ({pos_rate:.3f})")

    # Chosen threshold operating point
    _mark_threshold(
        ax,
        x=m_chosen["recall"],
        y=m_chosen["precision"],
        label=f"Chosen threshold ({chosen:.3f})\nP={m_chosen['precision']:.3f}  R={m_chosen['recall']:.3f}",
        color=RED,
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved {output_path}")


# ── plot 2: Threshold vs Precision / Recall ───────────────────────────────────

def plot_threshold_curve(pr: pd.DataFrame, metrics: dict, output_path: str) -> None:
    chosen = metrics["chosen_threshold"]
    m_chosen = metrics["metrics_at_chosen"]

    thresholds = pr["threshold"].values
    precisions = pr["precision"].values
    recalls    = pr["recall"].values

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(thresholds, precisions, color=BLUE,   lw=2, label="Precision")
    ax.plot(thresholds, recalls,    color=ORANGE,  lw=2, label="Recall")

    # Chosen threshold vertical line
    ax.axvline(chosen, color=RED, lw=1.4, linestyle="--",
               label=f"Chosen threshold ({chosen:.3f})")

    # Mark actual operating points
    ax.scatter([chosen], [m_chosen["precision"]], color=BLUE,   s=70, zorder=5)
    ax.scatter([chosen], [m_chosen["recall"]],    color=ORANGE, s=70, zorder=5)

    # Annotation box
    txt = (f"At threshold {chosen:.3f}\n"
           f"Precision = {m_chosen['precision']:.3f}\n"
           f"Recall      = {m_chosen['recall']:.3f}")
    ax.text(0.97, 0.97, txt, transform=ax.transAxes,
            fontsize=8.5, verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=GRAY, alpha=0.9))

    ax.set_xlabel("Classification Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold vs Precision / Recall")
    ax.set_xlim(thresholds.min(), thresholds.max())
    ax.set_ylim(0, 1)
    ax.legend(loc="center left", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved {output_path}")


# ── plot 3: ROC curve ─────────────────────────────────────────────────────────

def plot_roc_curve(roc: pd.DataFrame, metrics: dict, output_path: str) -> None:
    auc     = metrics["auc_roc"]
    chosen  = metrics["chosen_threshold"]
    m_chosen = metrics["metrics_at_chosen"]

    # Compute FPR at chosen threshold from confusion matrix
    cm = m_chosen["confusion_matrix"]
    fp, tn = cm["fp"], cm["tn"]
    fpr_at_chosen = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tpr_at_chosen = m_chosen["recall"]

    fig, ax = plt.subplots(figsize=(6, 6))

    # Random classifier diagonal
    ax.plot([0, 1], [0, 1], color=GRAY, lw=1.2, linestyle="--",
            label="Random classifier  (AUC = 0.50)")

    # ROC curve
    ax.plot(roc["fpr"], roc["tpr"], color=GREEN, lw=2,
            label=f"Model  (AUC = {auc:.3f})")

    # Chosen threshold operating point
    _mark_threshold(
        ax,
        x=fpr_at_chosen,
        y=tpr_at_chosen,
        label=f"Chosen threshold ({chosen:.3f})\nFPR={fpr_at_chosen:.3f}  TPR={tpr_at_chosen:.3f}",
        color=RED,
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate  (Recall)")
    ax.set_title("ROC Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved {output_path}")


# ── entry point ───────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    output_dir = args.output_dir or os.path.join(args.results_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading results from: {args.results_dir}")
    pr, roc, metrics = load_data(args.results_dir)

    print(f"Generating plots → {output_dir}")
    plot_pr_curve(
        pr, metrics,
        os.path.join(output_dir, "pr_curve.png"),
    )
    plot_threshold_curve(
        pr, metrics,
        os.path.join(output_dir, "threshold_curve.png"),
    )
    plot_roc_curve(
        roc, metrics,
        os.path.join(output_dir, "roc_curve.png"),
    )
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate PR, threshold, and ROC plots from evaluate_model.py outputs."
    )
    parser.add_argument(
        "--results_dir", default=DEFAULT_RESULTS,
        help="Directory containing pr_curve.csv, roc_curve.csv, metrics.json",
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Where to write plots (default: <results_dir>/plots/)",
    )
    main(parser.parse_args())
