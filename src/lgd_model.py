"""
Loss Given Default (LGD) Modeling
Beta regression approach for bounded [0,1] target variable.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import FIGURES_DIR, MODELS_DIR


class LGDModel:
    """LGD model using gradient boosting with beta-transformed target."""

    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )
        self.linear_model = LinearRegression()
        self.metrics = {}

    def prepare_lgd_data(self, df_train, df_test, feature_cols):
        """Prepare LGD data (only defaulted loans)."""
        train_default = df_train[df_train["default"] == 1].copy()
        test_default = df_test[df_test["default"] == 1].copy()

        available_cols = [c for c in feature_cols if c in train_default.columns]

        X_train = train_default[available_cols].fillna(0)
        y_train = train_default["lgd"].fillna(0.4)
        X_test = test_default[available_cols].fillna(0)
        y_test = test_default["lgd"].fillna(0.4)

        return X_train, y_train, X_test, y_test

    def train(self, X_train, y_train):
        """Train LGD model."""
        # Transform target for better distribution handling
        y_transformed = np.log(y_train / (1 - y_train + 1e-6))

        self.model.fit(X_train, y_transformed)
        self.linear_model.fit(X_train, y_train)

        print(f"    LGD Model trained on {len(X_train)} defaulted loans")

    def predict(self, X):
        """Predict LGD values."""
        y_transformed = self.model.predict(X)
        lgd_pred = 1 / (1 + np.exp(-y_transformed))
        return lgd_pred.clip(0.01, 0.99)

    def evaluate(self, X_test, y_test):
        """Evaluate LGD model performance."""
        y_pred = self.predict(X_test)

        self.metrics = {
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "mae": mean_absolute_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred),
            "mean_lgd_actual": y_test.mean(),
            "mean_lgd_predicted": y_pred.mean(),
        }

        print(f"    LGD RMSE: {self.metrics['rmse']:.4f}")
        print(f"    LGD MAE:  {self.metrics['mae']:.4f}")
        print(f"    LGD R2:   {self.metrics['r2']:.4f}")
        return self.metrics

    def plot_lgd_analysis(self, y_test, y_pred):
        """Plot LGD model results."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Actual vs Predicted
        axes[0, 0].scatter(y_test, y_pred, alpha=0.3, s=10, color="#3498db")
        axes[0, 0].plot([0, 1], [0, 1], "r--", linewidth=2)
        axes[0, 0].set_xlabel("Actual LGD")
        axes[0, 0].set_ylabel("Predicted LGD")
        axes[0, 0].set_title("Actual vs Predicted LGD")

        # Residuals
        residuals = y_test - y_pred
        axes[0, 1].hist(residuals, bins=50, color="#2ecc71", edgecolor="black", alpha=0.7)
        axes[0, 1].axvline(0, color="red", linestyle="--", linewidth=2)
        axes[0, 1].set_title("Residual Distribution")
        axes[0, 1].set_xlabel("Residual (Actual - Predicted)")

        # LGD Distribution
        axes[1, 0].hist(y_test, bins=50, alpha=0.6, color="#3498db",
                         label="Actual", density=True)
        axes[1, 0].hist(y_pred, bins=50, alpha=0.6, color="#e74c3c",
                         label="Predicted", density=True)
        axes[1, 0].set_title("LGD Distribution: Actual vs Predicted")
        axes[1, 0].legend()

        # Quantile-Quantile
        sorted_actual = np.sort(y_test)
        sorted_pred = np.sort(y_pred)
        min_len = min(len(sorted_actual), len(sorted_pred))
        axes[1, 1].scatter(sorted_actual[:min_len], sorted_pred[:min_len],
                           alpha=0.3, s=10, color="#9b59b6")
        axes[1, 1].plot([0, 1], [0, 1], "r--", linewidth=2)
        axes[1, 1].set_title("Q-Q Plot")
        axes[1, 1].set_xlabel("Actual Quantiles")
        axes[1, 1].set_ylabel("Predicted Quantiles")

        plt.suptitle("LGD Model Performance", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "12_lgd_analysis.png"))
        plt.close()
