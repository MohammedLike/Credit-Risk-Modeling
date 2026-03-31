"""
Basel II/III Regulatory Capital Calculations
IRB (Internal Ratings-Based) approach for credit risk capital.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import norm
from config import FIGURES_DIR, CONFIDENCE_LEVEL, MATURITY, LGD_DOWNTURN_MULTIPLIER


class BaselCapitalCalculator:
    """Basel II/III IRB capital requirements calculator."""

    def __init__(self):
        self.results = {}

    def asset_correlation(self, pd_val):
        """Basel II asset correlation function for corporate exposures."""
        r = (
            0.12 * (1 - np.exp(-50 * pd_val)) / (1 - np.exp(-50))
            + 0.24 * (1 - (1 - np.exp(-50 * pd_val)) / (1 - np.exp(-50)))
        )
        return r

    def maturity_adjustment(self, pd_val, m=MATURITY):
        """Basel II maturity adjustment factor."""
        b = (0.11852 - 0.05478 * np.log(pd_val)) ** 2
        ma = (1 + (m - 2.5) * b) / (1 - 1.5 * b)
        return ma

    def compute_capital_requirement(self, pd_val, lgd, ead, m=MATURITY):
        """Compute risk-weighted assets and capital per Basel IRB formula."""
        pd_val = np.clip(pd_val, 0.0003, 0.9999)

        R = self.asset_correlation(pd_val)
        MA = self.maturity_adjustment(pd_val, m)

        # Conditional PD under systemic stress (Vasicek model)
        conditional_pd = norm.cdf(
            (norm.ppf(pd_val) + np.sqrt(R) * norm.ppf(CONFIDENCE_LEVEL))
            / np.sqrt(1 - R)
        )

        # Capital requirement (K)
        K = (lgd * conditional_pd - pd_val * lgd) * MA
        K = np.maximum(K, 0)

        # Risk-Weighted Assets
        RWA = K * 12.5 * ead

        return {
            "capital_requirement_K": K,
            "rwa": RWA,
            "conditional_pd": conditional_pd,
            "asset_correlation": R,
            "maturity_adjustment": MA,
        }

    def compute_portfolio_capital(self, pd_array, lgd_array, ead_array):
        """Compute portfolio-level capital metrics."""
        results = self.compute_capital_requirement(pd_array, lgd_array, ead_array)

        total_ead = np.sum(ead_array)
        total_rwa = np.sum(results["rwa"])
        total_el = np.sum(pd_array * lgd_array * ead_array)
        total_capital = np.sum(results["capital_requirement_K"] * ead_array)

        # Minimum capital ratios (Basel III)
        cet1_requirement = total_rwa * 0.045  # 4.5% CET1
        tier1_requirement = total_rwa * 0.06  # 6% Tier 1
        total_capital_requirement = total_rwa * 0.08  # 8% Total
        conservation_buffer = total_rwa * 0.025  # 2.5% buffer

        self.results = {
            "total_ead": total_ead,
            "total_rwa": total_rwa,
            "rwa_density": total_rwa / total_ead,
            "total_el": total_el,
            "el_ratio": total_el / total_ead,
            "total_capital": total_capital,
            "capital_ratio": total_capital / total_ead,
            "avg_pd": pd_array.mean(),
            "avg_lgd": lgd_array.mean(),
            "avg_ead": ead_array.mean(),
            "cet1_requirement": cet1_requirement,
            "tier1_requirement": tier1_requirement,
            "total_capital_requirement": total_capital_requirement,
            "conservation_buffer": conservation_buffer,
            "rwa_per_exposure": results["rwa"],
            "capital_per_exposure": results["capital_requirement_K"] * ead_array,
        }

        return self.results

    def plot_capital_analysis(self, pd_array, lgd_array, ead_array):
        """Plot regulatory capital analysis."""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))

        # 1. RWA distribution
        rwa = self.results["rwa_per_exposure"]
        axes[0, 0].hist(rwa / 1000, bins=50, color="#3498db", edgecolor="black", alpha=0.7)
        axes[0, 0].set_title("RWA Distribution")
        axes[0, 0].set_xlabel("RWA ($K)")
        axes[0, 0].set_ylabel("Frequency")

        # 2. Capital K vs PD
        pd_range = np.linspace(0.001, 0.30, 100)
        lgd_val = 0.45
        K_values = []
        for pd_v in pd_range:
            res = self.compute_capital_requirement(pd_v, lgd_val, 1.0)
            K_values.append(res["capital_requirement_K"])
        axes[0, 1].plot(pd_range * 100, K_values, color="#e74c3c", linewidth=2)
        axes[0, 1].set_title("Capital Requirement vs PD (LGD=45%)")
        axes[0, 1].set_xlabel("PD (%)")
        axes[0, 1].set_ylabel("Capital Requirement (K)")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Asset correlation vs PD
        R_values = [self.asset_correlation(pd_v) for pd_v in pd_range]
        axes[0, 2].plot(pd_range * 100, R_values, color="#9b59b6", linewidth=2)
        axes[0, 2].set_title("Basel Asset Correlation vs PD")
        axes[0, 2].set_xlabel("PD (%)")
        axes[0, 2].set_ylabel("Asset Correlation (R)")
        axes[0, 2].grid(True, alpha=0.3)

        # 4. Capital waterfall
        res = self.results
        categories = ["CET1\n(4.5%)", "Add. Tier1\n(1.5%)", "Tier2\n(2%)",
                       "Conservation\nBuffer (2.5%)"]
        values = [
            res["cet1_requirement"] / 1e6,
            (res["tier1_requirement"] - res["cet1_requirement"]) / 1e6,
            (res["total_capital_requirement"] - res["tier1_requirement"]) / 1e6,
            res["conservation_buffer"] / 1e6,
        ]
        colors = ["#2c3e50", "#34495e", "#7f8c8d", "#bdc3c7"]
        axes[1, 0].bar(categories, values, color=colors, edgecolor="black")
        axes[1, 0].set_title("Capital Requirements Breakdown")
        axes[1, 0].set_ylabel("Capital ($M)")
        for i, v in enumerate(values):
            axes[1, 0].text(i, v + 0.5, f"${v:.1f}M", ha="center", fontweight="bold")

        # 5. PD vs EL contribution
        el_per_loan = pd_array * lgd_array * ead_array
        pd_bins = pd.qcut(pd_array, q=10, duplicates="drop")
        el_by_pd = pd.DataFrame({"pd_bin": pd_bins, "el": el_per_loan}).groupby(
            "pd_bin", observed=True
        )["el"].sum() / 1e6
        axes[1, 1].bar(range(len(el_by_pd)), el_by_pd.values,
                        color="#e67e22", edgecolor="black")
        axes[1, 1].set_title("Expected Loss by PD Decile")
        axes[1, 1].set_xlabel("PD Decile (Low to High)")
        axes[1, 1].set_ylabel("Expected Loss ($M)")

        # 6. Summary metrics
        axes[1, 2].axis("off")
        summary_text = (
            f"Portfolio Summary\n"
            f"{'=' * 35}\n"
            f"Total Exposure:  ${res['total_ead'] / 1e6:.1f}M\n"
            f"Total RWA:       ${res['total_rwa'] / 1e6:.1f}M\n"
            f"RWA Density:     {res['rwa_density']:.2%}\n"
            f"Expected Loss:   ${res['total_el'] / 1e6:.2f}M\n"
            f"EL Ratio:        {res['el_ratio']:.4%}\n"
            f"Total Capital:   ${res['total_capital'] / 1e6:.2f}M\n"
            f"Capital Ratio:   {res['capital_ratio']:.2%}\n"
            f"{'=' * 35}\n"
            f"Avg PD:   {res['avg_pd']:.4%}\n"
            f"Avg LGD:  {res['avg_lgd']:.4%}\n"
            f"Avg EAD:  ${res['avg_ead']:,.0f}\n"
        )
        axes[1, 2].text(0.1, 0.5, summary_text, fontfamily="monospace",
                         fontsize=12, verticalalignment="center",
                         bbox=dict(boxstyle="round,pad=0.5", facecolor="#ecf0f1"))

        plt.suptitle("Basel III Regulatory Capital Analysis",
                     fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "15_capital_analysis.png"))
        plt.close()
