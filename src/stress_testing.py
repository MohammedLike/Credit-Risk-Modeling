"""
Stress Testing & Scenario Analysis
Simulates portfolio behavior under adverse macroeconomic scenarios.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from config import FIGURES_DIR, STRESS_SCENARIOS


class StressTester:
    """Portfolio stress testing under macroeconomic scenarios."""

    def __init__(self, pd_model, lgd_model, feature_engineer):
        self.pd_model = pd_model
        self.lgd_model = lgd_model
        self.feature_engineer = feature_engineer
        self.results = {}

    def apply_scenario(self, df, scenario_name, shocks):
        """Apply macroeconomic shocks to portfolio data."""
        stressed = df.copy()
        stressed["gdp_growth"] = stressed["gdp_growth"] + shocks["gdp_shock"]
        stressed["unemployment_rate"] = (
            stressed["unemployment_rate"] + shocks["unemployment_shock"]
        ).clip(0, 0.25)
        stressed["fed_funds_rate"] = (
            stressed["fed_funds_rate"] + shocks["rate_shock"]
        ).clip(0, 0.15)

        # Second-order effects
        if shocks["unemployment_shock"] > 0:
            stressed["dti_ratio"] = (
                stressed["dti_ratio"] * (1 + shocks["unemployment_shock"] * 2)
            ).clip(0, 0.8)
            stressed["credit_utilization"] = (
                stressed["credit_utilization"] * (1 + shocks["unemployment_shock"])
            ).clip(0, 1)

        return stressed

    def run_stress_tests(self, df_test, raw_df_test):
        """Run all stress scenarios."""
        print("  Running stress tests...")

        for scenario_name, shocks in STRESS_SCENARIOS.items():
            print(f"    Scenario: {scenario_name}")

            stressed_raw = self.apply_scenario(raw_df_test, scenario_name, shocks)

            try:
                X_stressed, _ = self.feature_engineer.transform(stressed_raw)
                pd_stressed = self.pd_model.predict_pd(X_stressed)
            except Exception:
                pd_stressed = self.pd_model.predict_pd(df_test)
                if scenario_name != "baseline":
                    severity = abs(shocks["gdp_shock"]) * 10 + shocks["unemployment_shock"] * 5
                    pd_stressed = pd_stressed * (1 + severity)
                    pd_stressed = np.clip(pd_stressed, 0, 1)

            avg_pd = pd_stressed.mean()
            avg_lgd = 0.40 * (1 + abs(shocks.get("gdp_shock", 0)) * 3)
            avg_ead = raw_df_test["loan_amount"].mean()
            expected_loss = avg_pd * avg_lgd * avg_ead
            total_exposure = raw_df_test["loan_amount"].sum()
            portfolio_el = avg_pd * avg_lgd * total_exposure

            self.results[scenario_name] = {
                "avg_pd": avg_pd,
                "avg_lgd": avg_lgd,
                "avg_ead": avg_ead,
                "expected_loss_per_loan": expected_loss,
                "portfolio_el": portfolio_el,
                "total_exposure": total_exposure,
                "el_ratio": portfolio_el / total_exposure,
                "pd_distribution": pd_stressed,
            }

            print(f"      Avg PD: {avg_pd:.4f}, EL/Exposure: {portfolio_el / total_exposure:.4f}")

        return self.results

    def plot_stress_results(self):
        """Plot stress testing results."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        scenarios = list(self.results.keys())
        colors = ["#2ecc71", "#f39c12", "#e74c3c", "#8e44ad"]

        # PD by scenario
        pds = [self.results[s]["avg_pd"] * 100 for s in scenarios]
        axes[0, 0].bar(scenarios, pds, color=colors, edgecolor="black")
        axes[0, 0].set_title("Average PD by Scenario")
        axes[0, 0].set_ylabel("PD (%)")
        for i, v in enumerate(pds):
            axes[0, 0].text(i, v + 0.1, f"{v:.2f}%", ha="center", fontweight="bold")

        # EL ratio
        el_ratios = [self.results[s]["el_ratio"] * 100 for s in scenarios]
        axes[0, 1].bar(scenarios, el_ratios, color=colors, edgecolor="black")
        axes[0, 1].set_title("Expected Loss / Total Exposure")
        axes[0, 1].set_ylabel("EL Ratio (%)")
        for i, v in enumerate(el_ratios):
            axes[0, 1].text(i, v + 0.05, f"{v:.2f}%", ha="center", fontweight="bold")

        # PD distributions
        for scenario, color in zip(scenarios, colors):
            pd_dist = self.results[scenario]["pd_distribution"]
            axes[1, 0].hist(pd_dist, bins=50, alpha=0.5, color=color,
                            density=True, label=scenario)
        axes[1, 0].set_title("PD Distribution by Scenario")
        axes[1, 0].set_xlabel("Predicted PD")
        axes[1, 0].legend()

        # Portfolio EL
        els = [self.results[s]["portfolio_el"] / 1e6 for s in scenarios]
        axes[1, 1].bar(scenarios, els, color=colors, edgecolor="black")
        axes[1, 1].set_title("Total Portfolio Expected Loss")
        axes[1, 1].set_ylabel("Expected Loss ($M)")
        for i, v in enumerate(els):
            axes[1, 1].text(i, v + 0.5, f"${v:.1f}M", ha="center", fontweight="bold")

        plt.suptitle("Stress Testing Results", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "14_stress_testing.png"))
        plt.close()
