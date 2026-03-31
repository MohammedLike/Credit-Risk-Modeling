"""
Model Validation & Diagnostics
Implements ROC, KS, Gini, PSI, calibration, and discrimination tests.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, brier_score_loss,
    log_loss,
)
from sklearn.calibration import calibration_curve
from config import FIGURES_DIR


def compute_ks_statistic(y_true, y_pred):
    """Compute Kolmogorov-Smirnov statistic."""
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    ks = np.max(tpr - fpr)
    return ks


def compute_gini(y_true, y_pred):
    """Compute Gini coefficient = 2 * AUC - 1."""
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    return 2 * auc(fpr, tpr) - 1


def compute_psi(expected, actual, n_bins=10):
    """Compute Population Stability Index."""
    breakpoints = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_bins = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_bins = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    expected_bins = np.clip(expected_bins, 1e-6, None)
    actual_bins = np.clip(actual_bins, 1e-6, None)

    psi = np.sum((actual_bins - expected_bins) * np.log(actual_bins / expected_bins))
    return psi


def full_validation(y_train, y_train_pred, y_test, y_test_pred, model_name="Best Model"):
    """Run full model validation suite."""
    results = {}

    # ROC AUC
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_pred)
    auc_train = auc(fpr_train, tpr_train)
    auc_test = auc(fpr_test, tpr_test)

    results["auc_train"] = auc_train
    results["auc_test"] = auc_test
    results["ks_train"] = compute_ks_statistic(y_train, y_train_pred)
    results["ks_test"] = compute_ks_statistic(y_test, y_test_pred)
    results["gini_train"] = compute_gini(y_train, y_train_pred)
    results["gini_test"] = compute_gini(y_test, y_test_pred)
    results["brier_score"] = brier_score_loss(y_test, y_test_pred)
    results["log_loss"] = log_loss(y_test, y_test_pred)
    results["psi"] = compute_psi(y_train_pred, y_test_pred)

    # --- PLOTS ---
    fig = plt.figure(figsize=(20, 20))

    # 1. ROC Curve
    ax1 = fig.add_subplot(3, 3, 1)
    ax1.plot(fpr_train, tpr_train, color="#3498db", lw=2,
             label=f"Train AUC={auc_train:.4f}")
    ax1.plot(fpr_test, tpr_test, color="#e74c3c", lw=2,
             label=f"Test AUC={auc_test:.4f}")
    ax1.plot([0, 1], [0, 1], "k--", lw=1)
    ax1.set_title("ROC Curve")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend()

    # 2. Precision-Recall Curve
    ax2 = fig.add_subplot(3, 3, 2)
    prec, rec, _ = precision_recall_curve(y_test, y_test_pred)
    ap = average_precision_score(y_test, y_test_pred)
    ax2.plot(rec, prec, color="#2ecc71", lw=2, label=f"AP={ap:.4f}")
    ax2.set_title("Precision-Recall Curve")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend()

    # 3. KS Chart
    ax3 = fig.add_subplot(3, 3, 3)
    thresholds = np.linspace(0, 1, 100)
    tpr_ks = []
    fpr_ks = []
    for t in thresholds:
        tp = np.sum((y_test_pred >= t) & (y_test == 1))
        fp = np.sum((y_test_pred >= t) & (y_test == 0))
        tpr_ks.append(tp / np.sum(y_test == 1))
        fpr_ks.append(fp / np.sum(y_test == 0))
    tpr_ks = np.array(tpr_ks)
    fpr_ks = np.array(fpr_ks)
    ks_diff = tpr_ks - fpr_ks
    ks_idx = np.argmax(ks_diff)

    ax3.plot(thresholds, tpr_ks, label="TPR (Defaults)", color="#e74c3c")
    ax3.plot(thresholds, fpr_ks, label="FPR (Non-Defaults)", color="#3498db")
    ax3.axvline(thresholds[ks_idx], color="gray", linestyle="--",
                label=f"KS={ks_diff[ks_idx]:.4f}")
    ax3.fill_between(thresholds, fpr_ks, tpr_ks, alpha=0.1, color="green")
    ax3.set_title("KS Statistic Chart")
    ax3.set_xlabel("Threshold")
    ax3.legend()

    # 4. Calibration Plot
    ax4 = fig.add_subplot(3, 3, 4)
    fraction_pos, mean_pred = calibration_curve(y_test, y_test_pred, n_bins=10)
    ax4.plot(mean_pred, fraction_pos, "s-", color="#9b59b6", label="Model")
    ax4.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    ax4.set_title("Calibration Plot")
    ax4.set_xlabel("Mean Predicted Probability")
    ax4.set_ylabel("Fraction of Positives")
    ax4.legend()

    # 5. Score Distribution
    ax5 = fig.add_subplot(3, 3, 5)
    ax5.hist(y_test_pred[y_test == 0], bins=50, alpha=0.6, color="#2ecc71",
             density=True, label="Non-Default")
    ax5.hist(y_test_pred[y_test == 1], bins=50, alpha=0.6, color="#e74c3c",
             density=True, label="Default")
    ax5.set_title("PD Score Distribution")
    ax5.set_xlabel("Predicted PD")
    ax5.legend()

    # 6. Confusion Matrix (at optimal threshold)
    ax6 = fig.add_subplot(3, 3, 6)
    optimal_threshold = thresholds[ks_idx]
    y_pred_binary = (y_test_pred >= optimal_threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred_binary)
    im = ax6.imshow(cm, cmap="Blues")
    ax6.set_title(f"Confusion Matrix (threshold={optimal_threshold:.3f})")
    for i in range(2):
        for j in range(2):
            ax6.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                     fontsize=14, fontweight="bold",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax6.set_xlabel("Predicted")
    ax6.set_ylabel("Actual")
    ax6.set_xticks([0, 1])
    ax6.set_yticks([0, 1])
    ax6.set_xticklabels(["Non-Default", "Default"])
    ax6.set_yticklabels(["Non-Default", "Default"])

    # 7. Cumulative Gains
    ax7 = fig.add_subplot(3, 3, 7)
    sorted_idx = np.argsort(-y_test_pred)
    sorted_y = np.array(y_test)[sorted_idx]
    cum_defaults = np.cumsum(sorted_y) / np.sum(sorted_y)
    population_pct = np.arange(1, len(sorted_y) + 1) / len(sorted_y)
    ax7.plot(population_pct, cum_defaults, color="#e67e22", lw=2, label="Model")
    ax7.plot([0, 1], [0, 1], "k--", label="Random")
    ax7.set_title("Cumulative Gains Chart")
    ax7.set_xlabel("% Population")
    ax7.set_ylabel("% Defaults Captured")
    ax7.legend()

    # 8. Lift Chart
    ax8 = fig.add_subplot(3, 3, 8)
    n_bins = 10
    bin_size = len(sorted_y) // n_bins
    lift_values = []
    for i in range(n_bins):
        start = i * bin_size
        end = (i + 1) * bin_size if i < n_bins - 1 else len(sorted_y)
        bin_default_rate = sorted_y[start:end].mean()
        lift = bin_default_rate / y_test.mean()
        lift_values.append(lift)
    ax8.bar(range(1, n_bins + 1), lift_values, color="#1abc9c", edgecolor="black")
    ax8.axhline(1.0, color="red", linestyle="--", label="Baseline")
    ax8.set_title("Lift Chart (Decile)")
    ax8.set_xlabel("Decile (High Risk -> Low Risk)")
    ax8.set_ylabel("Lift")
    ax8.legend()

    # 9. PSI Distribution
    ax9 = fig.add_subplot(3, 3, 9)
    n_psi_bins = 10
    breakpoints = np.percentile(y_train_pred, np.linspace(0, 100, n_psi_bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    train_dist = np.histogram(y_train_pred, bins=breakpoints)[0] / len(y_train_pred)
    test_dist = np.histogram(y_test_pred, bins=breakpoints)[0] / len(y_test_pred)
    x_pos = np.arange(n_psi_bins)
    width = 0.35
    ax9.bar(x_pos - width / 2, train_dist, width, label="Train", color="#3498db", edgecolor="black")
    ax9.bar(x_pos + width / 2, test_dist, width, label="Test", color="#e74c3c", edgecolor="black")
    ax9.set_title(f"PSI Analysis (PSI={results['psi']:.4f})")
    ax9.set_xlabel("Score Bin")
    ax9.set_ylabel("Proportion")
    ax9.legend()

    plt.suptitle(f"Model Validation Dashboard — {model_name}",
                 fontsize=18, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "09_validation_dashboard.png"))
    plt.close()

    return results


def plot_decile_analysis(y_test, y_test_pred):
    """Detailed decile analysis."""
    df = pd.DataFrame({"actual": y_test, "predicted": y_test_pred})
    df["decile"] = pd.qcut(df["predicted"], q=10, labels=False, duplicates="drop") + 1

    decile_stats = df.groupby("decile").agg(
        count=("actual", "count"),
        default_rate=("actual", "mean"),
        avg_pd=("predicted", "mean"),
        min_pd=("predicted", "min"),
        max_pd=("predicted", "max"),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].bar(decile_stats["decile"], decile_stats["default_rate"] * 100,
                color="#e74c3c", edgecolor="black", alpha=0.7, label="Actual Default Rate")
    axes[0].bar(decile_stats["decile"], decile_stats["avg_pd"] * 100,
                color="#3498db", edgecolor="black", alpha=0.5, label="Avg Predicted PD")
    axes[0].set_title("Default Rate by Risk Decile")
    axes[0].set_xlabel("Risk Decile (1=Lowest Risk)")
    axes[0].set_ylabel("Rate (%)")
    axes[0].legend()

    axes[1].plot(decile_stats["decile"], decile_stats["default_rate"],
                 "ro-", label="Actual", linewidth=2)
    axes[1].plot(decile_stats["decile"], decile_stats["avg_pd"],
                 "bs-", label="Predicted", linewidth=2)
    axes[1].set_title("Calibration by Decile")
    axes[1].set_xlabel("Risk Decile")
    axes[1].set_ylabel("Default Rate")
    axes[1].legend()

    plt.suptitle("Risk Decile Analysis", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "10_decile_analysis.png"))
    plt.close()

    return decile_stats
