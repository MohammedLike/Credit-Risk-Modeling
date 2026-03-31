"""
Exposure at Default (EAD) Modeling
Models the Credit Conversion Factor (CCF) approach per Basel framework.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import FIGURES_DIR


class EADModel:
    """EAD model using Credit Conversion Factor (CCF) approach."""

    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )
        self.metrics = {}

    def prepare_ead_data(self, df_train, df_test, feature_cols):
        """Prepare EAD data — compute CCF for defaulted loans."""
        train_default = df_train[df_train["default"] == 1].copy()
        test_default = df_test[df_test["default"] == 1].copy()

        # CCF = (EAD - Drawn) / (Limit - Drawn)
        for data in [train_default, test_default]:
            undrawn = data["credit_limit"] - data["drawn_amount"]
            data["ccf"] = np.where(
                undrawn > 0,
                (data["ead"] - data["drawn_amount"]) / undrawn,
                0,
            ).clip(0, 1)

        available_cols = [c for c in feature_cols if c in train_default.columns]

        X_train = train_default[available_cols].fillna(0)
        y_train = train_default["ccf"]
        X_test = test_default[available_cols].fillna(0)
        y_test = test_default["ccf"]

        return X_train, y_train, X_test, y_test

    def train(self, X_train, y_train):
        """Train EAD/CCF model."""
        self.model.fit(X_train, y_train)
        print(f"    EAD Model trained on {len(X_train)} defaulted loans")

    def predict_ccf(self, X):
        """Predict Credit Conversion Factor."""
        return self.model.predict(X).clip(0, 1)

    def predict_ead(self, drawn, limit, X):
        """Predict EAD = Drawn + CCF * (Limit - Drawn)."""
        ccf = self.predict_ccf(X)
        return drawn + ccf * (limit - drawn)

    def evaluate(self, X_test, y_test):
        """Evaluate EAD model."""
        y_pred = self.predict_ccf(X_test)
        self.metrics = {
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "mae": mean_absolute_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred),
            "mean_ccf_actual": y_test.mean(),
            "mean_ccf_predicted": y_pred.mean(),
        }
        print(f"    EAD CCF RMSE: {self.metrics['rmse']:.4f}")
        print(f"    EAD CCF R2:   {self.metrics['r2']:.4f}")
        return self.metrics

    def plot_ead_analysis(self, y_test, y_pred):
        """Plot EAD model results."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].scatter(y_test, y_pred, alpha=0.3, s=10, color="#e67e22")
        axes[0].plot([0, 1], [0, 1], "r--", linewidth=2)
        axes[0].set_xlabel("Actual CCF")
        axes[0].set_ylabel("Predicted CCF")
        axes[0].set_title("Actual vs Predicted CCF")

        axes[1].hist(y_test, bins=40, alpha=0.6, color="#3498db", label="Actual", density=True)
        axes[1].hist(y_pred, bins=40, alpha=0.6, color="#e74c3c", label="Predicted", density=True)
        axes[1].set_title("CCF Distribution")
        axes[1].legend()

        residuals = y_test - y_pred
        axes[2].hist(residuals, bins=40, color="#2ecc71", edgecolor="black", alpha=0.7)
        axes[2].axvline(0, color="red", linestyle="--")
        axes[2].set_title("CCF Residuals")

        plt.suptitle("EAD/CCF Model Performance", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "13_ead_analysis.png"))
        plt.close()
