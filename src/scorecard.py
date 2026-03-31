"""
Credit Scorecard Development
Weight of Evidence (WoE) and Information Value (IV) based scorecard.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from config import FIGURES_DIR


class CreditScorecard:
    """WoE-based credit scorecard with points allocation."""

    BASE_SCORE = 600
    PDO = 20  # Points to Double the Odds
    TARGET_ODDS = 50  # odds at base score (50:1 good:bad)

    def __init__(self):
        self.woe_tables = {}
        self.iv_values = {}
        self.scorecard_points = {}
        self.factor = None
        self.offset = None

    def _compute_woe_iv(self, df, feature, target, n_bins=10):
        """Compute WoE and IV for a feature."""
        data = df[[feature, target]].copy()

        if data[feature].dtype in ["object", "category"]:
            data["bin"] = data[feature]
        else:
            data["bin"] = pd.qcut(data[feature], q=n_bins, duplicates="drop")

        grouped = data.groupby("bin", observed=True)[target].agg(["sum", "count"])
        grouped.columns = ["bad", "total"]
        grouped["good"] = grouped["total"] - grouped["bad"]

        grouped["bad_pct"] = grouped["bad"] / grouped["bad"].sum()
        grouped["good_pct"] = grouped["good"] / grouped["good"].sum()

        grouped["bad_pct"] = grouped["bad_pct"].clip(lower=0.0001)
        grouped["good_pct"] = grouped["good_pct"].clip(lower=0.0001)

        grouped["woe"] = np.log(grouped["good_pct"] / grouped["bad_pct"])
        grouped["iv"] = (grouped["good_pct"] - grouped["bad_pct"]) * grouped["woe"]

        total_iv = grouped["iv"].sum()
        return grouped, total_iv

    def fit(self, df, features, target="default"):
        """Build WoE tables and scorecard."""
        print("    Building WoE scorecard...")

        # Compute scaling factors
        self.factor = self.PDO / np.log(2)
        self.offset = self.BASE_SCORE - self.factor * np.log(self.TARGET_ODDS)

        for feat in features:
            try:
                woe_table, iv = self._compute_woe_iv(df, feat, target)
                self.woe_tables[feat] = woe_table
                self.iv_values[feat] = iv
            except Exception:
                continue

        # Sort by IV and keep top predictive features
        sorted_iv = sorted(self.iv_values.items(), key=lambda x: x[1], reverse=True)

        print(f"    Top features by IV:")
        for feat, iv in sorted_iv[:10]:
            predictive = "Highly" if iv > 0.3 else "Moderately" if iv > 0.1 else "Weakly"
            print(f"      {feat}: IV={iv:.4f} ({predictive} Predictive)")

        return sorted_iv

    def compute_score(self, pd_value):
        """Convert PD to credit score."""
        odds = (1 - pd_value) / (pd_value + 1e-10)
        score = self.offset + self.factor * np.log(odds)
        return np.clip(score, 300, 850).round(0)

    def assign_rating(self, score):
        """Map score to rating grade."""
        if score >= 750:
            return "AAA"
        elif score >= 720:
            return "AA"
        elif score >= 690:
            return "A"
        elif score >= 660:
            return "BBB"
        elif score >= 630:
            return "BB"
        elif score >= 600:
            return "B"
        elif score >= 550:
            return "CCC"
        else:
            return "D"

    def plot_scorecard_results(self, scores, y_true, pd_values):
        """Plot scorecard analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Score distribution
        axes[0, 0].hist(scores[y_true == 0], bins=50, alpha=0.6, color="#2ecc71",
                         density=True, label="Non-Default")
        axes[0, 0].hist(scores[y_true == 1], bins=50, alpha=0.6, color="#e74c3c",
                         density=True, label="Default")
        axes[0, 0].set_title("Credit Score Distribution")
        axes[0, 0].set_xlabel("Credit Score")
        axes[0, 0].legend()

        # Score vs Default Rate
        score_bins = pd.qcut(scores, q=20, duplicates="drop")
        df_temp = pd.DataFrame({"score_bin": score_bins, "default": y_true})
        default_by_score = df_temp.groupby("score_bin", observed=True)["default"].mean() * 100
        axes[0, 1].plot(range(len(default_by_score)), default_by_score.values,
                        "ro-", linewidth=2)
        axes[0, 1].set_title("Default Rate by Score Band")
        axes[0, 1].set_xlabel("Score Band (Low to High)")
        axes[0, 1].set_ylabel("Default Rate (%)")

        # IV Chart
        if self.iv_values:
            sorted_iv = sorted(self.iv_values.items(), key=lambda x: x[1], reverse=True)[:15]
            names, values = zip(*sorted_iv)
            colors = ["#e74c3c" if v > 0.3 else "#f39c12" if v > 0.1 else "#3498db"
                       for v in values]
            axes[1, 0].barh(names, values, color=colors, edgecolor="black")
            axes[1, 0].set_title("Information Value by Feature")
            axes[1, 0].set_xlabel("IV")
            axes[1, 0].axvline(0.1, color="gray", linestyle="--", alpha=0.5)
            axes[1, 0].axvline(0.3, color="gray", linestyle="--", alpha=0.5)

        # Rating Distribution
        ratings = pd.Series([self.assign_rating(s) for s in scores])
        rating_order = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D"]
        rating_counts = ratings.value_counts().reindex(rating_order).fillna(0)
        colors_rating = ["#27ae60", "#2ecc71", "#82e0aa", "#f9e79f",
                          "#f5b041", "#e74c3c", "#c0392b", "#7b241c"]
        axes[1, 1].bar(rating_counts.index, rating_counts.values,
                        color=colors_rating[:len(rating_counts)], edgecolor="black")
        axes[1, 1].set_title("Rating Distribution")
        axes[1, 1].set_xlabel("Rating Grade")
        axes[1, 1].set_ylabel("Count")

        plt.suptitle("Credit Scorecard Analysis", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "11_scorecard.png"))
        plt.close()
