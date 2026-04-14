"""
Probability of Default (PD) Modeling
Implements Logistic Regression, Random Forest, XGBoost, CatBoost, and ensemble methods.
Includes Optuna hyperparameter optimization.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import joblib
import optuna
from optuna.pruners import MedianPruner
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from config import FIGURES_DIR, MODELS_DIR, N_FOLDS


class PDModelSuite:
    """Suite of PD models with training, comparison, selection, and Optuna tuning."""

    def __init__(self, use_optuna=False):
        """
        Initialize PD model suite.
        
        Args:
            use_optuna: If True, use Optuna for hyperparameter optimization
        """
        self.use_optuna = use_optuna
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
                eval_metric="logloss", verbosity=0,
            ),
            "CatBoost": CatBoostClassifier(
                iterations=200, depth=6, learning_rate=0.05,
                l2_leaf_reg=5.0, subsample=0.8,
                random_state=42, verbose=0,
            ),
        }
        self.fitted_models = {}
        self.cv_results = {}
        self.best_model_name = None
        self.optuna_results = {}

    def train_all(self, X_train, y_train):
        """Train all models and perform cross-validation."""
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

        for name, model in self.models.items():
            print(f"    Training {name}...")
            
            # Optuna hyperparameter tuning for tree-based models
            if self.use_optuna and name in ["XGBoost", "CatBoost", "Random Forest"]:
                print(f"      Running Optuna optimization for {name}...")
                model = self._optuna_tune(X_train, y_train, name, skf)
                self.models[name] = model
            
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
    
    def _optuna_tune(self, X_train, y_train, model_name, skf, n_trials=50):
        """
        Optuna hyperparameter optimization.
        
        Args:
            X_train: Training features
            y_train: Training labels
            model_name: Name of model to tune
            skf: StratifiedKFold cross-validator
            n_trials: Number of trials
        
        Returns:
            Optimized model with best parameters
        """
        def objective(trial):
            if model_name == "XGBoost":
                params = {
                    'max_depth': trial.suggest_int('xgb_max_depth', 3, 8),
                    'learning_rate': trial.suggest_float('xgb_lr', 0.01, 0.3, log=True),
                    'n_estimators': trial.suggest_int('xgb_n_est', 100, 400, step=50),
                    'subsample': trial.suggest_float('xgb_subsample', 0.5, 1.0),
                    'reg_alpha': trial.suggest_float('xgb_alpha', 0.0, 1.0),
                    'reg_lambda': trial.suggest_float('xgb_lambda', 0.0, 1.0),
                }
                model = XGBClassifier(**params, random_state=42, use_label_encoder=False, 
                                     eval_metric="logloss", verbosity=0)
            
            elif model_name == "CatBoost":
                params = {
                    'depth': trial.suggest_int('cat_depth', 4, 8),
                    'learning_rate': trial.suggest_float('cat_lr', 0.01, 0.3, log=True),
                    'iterations': trial.suggest_int('cat_iter', 100, 300, step=50),
                    'l2_leaf_reg': trial.suggest_float('cat_l2', 1.0, 10.0),
                    'subsample': trial.suggest_float('cat_subsample', 0.5, 1.0),
                }
                model = CatBoostClassifier(**params, random_state=42, verbose=0)
            
            elif model_name == "Random Forest":
                params = {
                    'n_estimators': trial.suggest_int('rf_n_est', 100, 500, step=50),
                    'max_depth': trial.suggest_int('rf_max_depth', 5, 15),
                    'min_samples_leaf': trial.suggest_int('rf_min_leaf', 20, 100, step=10),
                }
                model = RandomForestClassifier(**params, random_state=42, 
                                              class_weight="balanced", n_jobs=-1)
            
            cv_scores = cross_val_score(model, X_train, y_train, cv=skf, 
                                       scoring="roc_auc", n_jobs=-1)
            return cv_scores.mean()
        
        sampler = optuna.samplers.TPESampler(seed=42)
        pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")
        
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        self.optuna_results[model_name] = {
            "best_params": study.best_params,
            "best_score": study.best_value,
        }
        
        best_params = study.best_params
        
        # Create model with best parameters
        if model_name == "XGBoost":
            return XGBClassifier(**best_params, random_state=42, 
                               use_label_encoder=False, eval_metric="logloss", verbosity=0)
        elif model_name == "CatBoost":
            return CatBoostClassifier(**best_params, random_state=42, verbose=0)
        elif model_name == "Random Forest":
            return RandomForestClassifier(**best_params, random_state=42, 
                                         class_weight="balanced", n_jobs=-1)

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

        colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"]
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
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        tree_models = [
            ("Random Forest", self.fitted_models.get("Random Forest")),
            ("XGBoost", self.fitted_models.get("XGBoost")),
            ("Gradient Boosting", self.fitted_models.get("Gradient Boosting")),
            ("CatBoost", self.fitted_models.get("CatBoost")),
        ]

        for ax, (name, model) in zip(axes, tree_models):
            if model is None:
                ax.text(0.5, 0.5, f"{name}\nNot available", ha="center", va="center")
                ax.set_title(f"{name} - Feature Importance")
                continue
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            else:
                continue
            
            indices = np.argsort(importances)[-top_n:]

            ax.barh(
                [feature_names[i] for i in indices],
                importances[indices],
                color="#3498db", edgecolor="black",
            )
            ax.set_title(f"{name} - Top {top_n} Features", fontweight="bold")
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
            if name in self.optuna_results:
                results[name]["optuna_best_score"] = round(self.optuna_results[name]["best_score"], 4)
        
        with open(os.path.join(MODELS_DIR, "pd_model_results.json"), "w") as f:
            json.dump(results, f, indent=2)
    
    def save_models(self):
        """Save all trained models to disk."""
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        for name, model in self.fitted_models.items():
            model_path = os.path.join(MODELS_DIR, f"pd_{name.lower().replace(' ', '_')}.pkl")
            joblib.dump(model, model_path)
            print(f"  Saved: {model_path}")
        
        # Save metadata
        metadata = {
            "best_model": self.best_model_name,
            "cv_results": self.cv_results,
            "optuna_results": self.optuna_results,
        }
        metadata_path = os.path.join(MODELS_DIR, "pd_model_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    @staticmethod
    def load_models(model_dir=MODELS_DIR):
        """
        Load previously trained models.
        
        Args:
            model_dir: Directory containing saved models
        
        Returns:
            Dictionary of loaded models
        """
        loaded_models = {}
        for file in os.listdir(model_dir):
            if file.startswith("pd_") and file.endswith(".pkl"):
                model_path = os.path.join(model_dir, file)
                model_name = file.replace("pd_", "").replace(".pkl", "").replace("_", " ").title()
                loaded_models[model_name] = joblib.load(model_path)
        
        return loaded_models
