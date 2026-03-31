"""
Exploratory Data Analysis for Credit Risk Dataset
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import FIGURES_DIR
import os


def set_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.dpi": 150,
    })


def plot_default_distribution(df):
    """Plot default rate distribution."""
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Count
    colors = ["#2ecc71", "#e74c3c"]
    df["default"].value_counts().sort_index().plot(
        kind="bar", ax=axes[0], color=colors, edgecolor="black"
    )
    axes[0].set_title("Default Distribution (Count)")
    axes[0].set_xticklabels(["Non-Default", "Default"], rotation=0)
    axes[0].set_ylabel("Count")
    for p in axes[0].patches:
        axes[0].annotate(
            f"{int(p.get_height()):,}",
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha="center", va="bottom", fontweight="bold",
        )

    # Percentage
    rates = df["default"].value_counts(normalize=True).sort_index() * 100
    rates.plot(kind="pie", ax=axes[1], colors=colors, autopct="%1.1f%%",
               startangle=90, labels=["Non-Default", "Default"])
    axes[1].set_title("Default Rate")
    axes[1].set_ylabel("")

    plt.suptitle("Target Variable: Loan Default", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "01_default_distribution.png"))
    plt.close()


def plot_feature_distributions(df):
    """Plot distributions of key numerical features by default status."""
    set_style()
    features = ["fico_score", "annual_income", "dti_ratio", "credit_utilization",
                 "loan_amount", "interest_rate", "employment_length", "revolving_balance"]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()

    for i, feat in enumerate(features):
        for label, color in zip([0, 1], ["#2ecc71", "#e74c3c"]):
            subset = df[df["default"] == label][feat].dropna()
            axes[i].hist(subset, bins=40, alpha=0.6, color=color, density=True,
                         label="Default" if label else "Non-Default")
        axes[i].set_title(feat.replace("_", " ").title())
        axes[i].legend(fontsize=8)

    plt.suptitle("Feature Distributions by Default Status", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "02_feature_distributions.png"))
    plt.close()


def plot_correlation_matrix(df):
    """Plot correlation heatmap of numerical features."""
    set_style()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    exclude = ["lgd", "ead", "credit_limit", "drawn_amount"]
    cols = [c for c in numeric_cols if c not in exclude]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, square=True, linewidths=0.5, ax=ax,
        cbar_kws={"shrink": 0.8}, annot_kws={"size": 7},
    )
    ax.set_title("Feature Correlation Matrix", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "03_correlation_matrix.png"))
    plt.close()


def plot_fico_vs_default(df):
    """Plot FICO score vs default rate."""
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # FICO distribution
    for label, color in zip([0, 1], ["#2ecc71", "#e74c3c"]):
        subset = df[df["default"] == label]["fico_score"]
        axes[0].hist(subset, bins=50, alpha=0.6, color=color, density=True,
                     label="Default" if label else "Non-Default")
    axes[0].set_title("FICO Score Distribution by Default")
    axes[0].set_xlabel("FICO Score")
    axes[0].legend()

    # Default rate by FICO bucket
    bins = list(range(300, 860, 25))
    df["fico_bin"] = pd.cut(df["fico_score"], bins=bins)
    default_by_fico = df.groupby("fico_bin", observed=True)["default"].mean() * 100
    default_by_fico.plot(kind="bar", ax=axes[1], color="#3498db", edgecolor="black")
    axes[1].set_title("Default Rate by FICO Score Band")
    axes[1].set_xlabel("FICO Score Range")
    axes[1].set_ylabel("Default Rate (%)")
    axes[1].tick_params(axis="x", rotation=45)
    df.drop(columns=["fico_bin"], inplace=True)

    plt.suptitle("FICO Score Analysis", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "04_fico_analysis.png"))
    plt.close()


def plot_macro_impact(df):
    """Plot macroeconomic feature impact on defaults."""
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    macro_features = [
        ("gdp_growth", "GDP Growth Rate"),
        ("unemployment_rate", "Unemployment Rate"),
        ("fed_funds_rate", "Federal Funds Rate"),
    ]

    for ax, (feat, title) in zip(axes, macro_features):
        bins = pd.qcut(df[feat], q=10, duplicates="drop")
        default_by_bin = df.groupby(bins, observed=True)["default"].mean() * 100
        default_by_bin.plot(kind="bar", ax=ax, color="#9b59b6", edgecolor="black")
        ax.set_title(f"Default Rate vs {title}")
        ax.set_xlabel(title)
        ax.set_ylabel("Default Rate (%)")
        ax.tick_params(axis="x", rotation=45)

    plt.suptitle("Macroeconomic Impact on Default Rates", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "05_macro_impact.png"))
    plt.close()


def plot_loan_purpose_analysis(df):
    """Analyze default rates by loan purpose."""
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    purpose_counts = df["loan_purpose"].value_counts()
    purpose_counts.plot(kind="barh", ax=axes[0], color="#3498db", edgecolor="black")
    axes[0].set_title("Loan Purpose Distribution")
    axes[0].set_xlabel("Count")

    purpose_default = df.groupby("loan_purpose")["default"].mean().sort_values() * 100
    purpose_default.plot(kind="barh", ax=axes[1], color="#e74c3c", edgecolor="black")
    axes[1].set_title("Default Rate by Loan Purpose")
    axes[1].set_xlabel("Default Rate (%)")

    plt.suptitle("Loan Purpose Analysis", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "06_loan_purpose.png"))
    plt.close()


def generate_summary_statistics(df):
    """Generate and save summary statistics."""
    stats = df.describe(include="all").round(3)
    stats.to_csv(os.path.join(FIGURES_DIR, "..", "data", "summary_statistics.csv"))

    default_summary = {
        "total_loans": len(df),
        "default_count": df["default"].sum(),
        "default_rate": df["default"].mean(),
        "avg_loan_amount": df["loan_amount"].mean(),
        "avg_fico": df["fico_score"].mean(),
        "avg_dti": df["dti_ratio"].mean(),
        "avg_interest_rate": df["interest_rate"].mean(),
        "avg_income": df["annual_income"].mean(),
    }
    return default_summary


def run_eda(df):
    """Run complete EDA pipeline."""
    print("  [EDA] Plotting default distribution...")
    plot_default_distribution(df)
    print("  [EDA] Plotting feature distributions...")
    plot_feature_distributions(df)
    print("  [EDA] Plotting correlation matrix...")
    plot_correlation_matrix(df)
    print("  [EDA] Plotting FICO analysis...")
    plot_fico_vs_default(df)
    print("  [EDA] Plotting macro impact...")
    plot_macro_impact(df)
    print("  [EDA] Plotting loan purpose analysis...")
    plot_loan_purpose_analysis(df)
    print("  [EDA] Generating summary statistics...")
    summary = generate_summary_statistics(df)
    print("  [EDA] Complete.")
    return summary
