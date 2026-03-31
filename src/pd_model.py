"""
Probability of Default (PD) Modeling
Implements Logistic Regression, Random Forest, XGBoost, and ensemble methods.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost import XGBClassifier
from config import FIGURES_DIR, MODELS_DIR, N_FOLDS


class PDModelSuite:
    """Suite of PD models with training, comparison, and selection."""

    def __init__(self):
        self.models = {
            "Logistic Regression": LogisticRegression(
                C=0.1, penalty="l2", solver="lbfgs",
                max_iter=1000, random_state=42, class_weight="balanced",
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=300, max_depth=8, min_samples_leaf=50,
                random_state=42, class_weight="balanced", n_jobs=-1,
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=42,
            ),
            "XGBoost": XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0,
                random_state=42, use_label_encoder=False,
                eval_metric="logloss",
            ),
        }
        self.fitted_models = {}
        self.cv_results = {}
        self.best_model_name = None

    def train_all(self, X_train, y_train):
        """Train all models and perform cross-validation."""
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

        for name, model in self.models.items():
            print(f"    Training {name}...")
            model.fit(X_train, y_train)
            self.fitted_models[name] = model

            # Cross-validation
            cv_auc = cross_val_score(
                model, X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1
            )
            self.cv_results[name] = {
                "mean_auc": cv_auc.mean(),
                "std_auc": cv_auc.std(),
                "cv_scores": cv_auc.tolist(),
            }
            print(f"      CV AUC: {cv_auc.mean():.4f} (+/- {cv_auc.std():.4f})")

        # Select best model
        self.best_model_name = max(
            self.cv_results, key=lambda k: self.cv_results[k]["mean_auc"]
        )
        print(f"    Best PD Model: {self.best_model_name}")
        return self.cv_results

    def predict_pd(self, X, model_name=None):
        """Predict probability of default."""
        name = model_name or self.best_model_name
        return self.fitted_models[name].predict_proba(X)[:, 1]

    def get_best_model(self):
        return self.fitted_models[self.best_model_name]

    def plot_model_comparison(self):
        """Plot cross-validation results comparison."""
        fig, ax = plt.subplots(figsize=(10, 6))

        names = list(self.cv_results.keys())
        means = [self.cv_results[n]["mean_auc"] for n in names]
        stds = [self.cv_results[n]["std_auc"] for n in names]

        colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12"]
        bars = ax.barh(names, means, xerr=stds, color=colors, edgecolor="black",
                       capsize=5, height=0.5)

        for bar, mean in zip(bars, means):
            ax.text(mean + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"{mean:.4f}", va="center", fontweight="bold")

        ax.set_xlabel("ROC AUC Score")
        ax.set_title("PD Model Comparison (5-Fold Cross-Validation)",
                      fontsize=14, fontweight="bold")
        ax.set_xlim(min(means) - 0.05, max(means) + 0.03)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "07_model_comparison.png"))
        plt.close()

    def plot_feature_importance(self, feature_names, top_n=20):
        """Plot feature importance for tree-based models."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        for ax, name in zip(axes, ["Random Forest", "XGBoost"]):
            model = self.fitted_models[name]
            importances = model.feature_importances_
            indices = np.argsort(importances)[-top_n:]

            ax.barh(
                [feature_names[i] for i in indices],
                importances[indices],
                color="#3498db", edgecolor="black",
            )
            ax.set_title(f"{name} - Top {top_n} Features")
            ax.set_xlabel("Feature Importance")

        plt.suptitle("Feature Importance Analysis", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "08_feature_importance.png"))
        plt.close()

    def save_results(self):
        """Save model comparison results."""
        results = {}
        for name, res in self.cv_results.items():
            results[name] = {
                "mean_auc": round(res["mean_auc"], 4),
                "std_auc": round(res["std_auc"], 4),
            }
        with open(os.path.join(MODELS_DIR, "pd_model_results.json"), "w") as f:
            json.dump(results, f, indent=2)
