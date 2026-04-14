"""
Model Explainability using SHAP and Partial Dependence
Provides interpretability for credit risk models (Basel compliance)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.inspection import partial_dependence
import os
from config import FIGURES_DIR, MODELS_DIR


class ModelExplainability:
    """SHAP-based model explainability for credit risk models."""

    def __init__(self, model, X_train, feature_names):
        """
        Initialize explainability analyzer.
        
        Args:
            model: Fitted sklearn/xgboost model
            X_train: Training data (for SHAP background)
            feature_names: List of feature names
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        
    def compute_shap_values(self, X, model_type=None):
        """
        Compute SHAP values for explanations.
        
        Args:
            X: Data to explain
            model_type: "tree", "linear", or "kernel" (auto-detect if None)
        """
        print("  Computing SHAP values...")
        
        # Auto-detect model type if not specified
        if model_type is None:
            model_class_name = self.model.__class__.__name__
            if 'LogisticRegression' in model_class_name or 'Linear' in model_class_name:
                model_type = "linear"
            elif any(x in model_class_name for x in ['Tree', 'Forest', 'XGB', 'CatBoost', 'Gradient']):
                model_type = "tree"
            else:
                model_type = "kernel"
        
        if model_type == "tree":
            try:
                self.explainer = shap.TreeExplainer(self.model)
            except Exception as e:
                print(f"    TreeExplainer failed ({e}), falling back to KernelExplainer...")
                model_type = "kernel"
        
        if model_type == "linear":
            try:
                self.explainer = shap.LinearExplainer(self.model, self.X_train)
            except Exception as e:
                print(f"    LinearExplainer failed ({e}), falling back to KernelExplainer...")
                model_type = "kernel"
        
        if model_type == "kernel":
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba, 
                shap.sample(self.X_train, min(100, len(self.X_train)))
            )
        
        self.shap_values = self.explainer.shap_values(X)
        
        # For binary classification, get probabilities for default class
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]
        
        return self.shap_values
    
    def plot_beeswarm(self, X, max_display=15, save_path=None):
        """
        Plot SHAP beeswarm chart (feature importance by impact).
        
        Args:
            X: Data to explain
            max_display: Number of features to display
            save_path: Path to save figure
        """
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        plt.figure(figsize=(12, 8))
        
        # Create DataFrame with SHAP values for easier handling
        shap.plots.beeswarm(
            shap.Explanation(
                values=self.shap_values,
                base_values=self.explainer.expected_value,
                data=X,
                feature_names=self.feature_names
            ),
            max_display=max_display,
            show=False
        )
        
        plt.title("SHAP Beeswarm Plot: Feature Impact on PD Predictions", 
                  fontsize=14, fontweight="bold")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"  Saved: {save_path}")
        
        plt.close()
    
    def plot_force_diagram(self, X, sample_idx=0, save_path=None):
        """
        Plot SHAP force diagram for individual predictions.
        
        Args:
            X: Data
            sample_idx: Index of sample to explain
            save_path: Path to save figure
        """
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        plt.figure(figsize=(14, 4))
        
        shap.plots.force(
            self.explainer.expected_value,
            self.shap_values[sample_idx],
            X.iloc[sample_idx],
            feature_names=self.feature_names,
            matplotlib=True,
            show=False
        )
        
        plt.title(f"SHAP Force Diagram: Loan {sample_idx}", 
                  fontsize=12, fontweight="bold")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"  Saved: {save_path}")
        
        plt.close()
    
    def plot_summary_bar(self, X, max_display=15, save_path=None):
        """
        Plot SHAP summary bar chart (mean absolute SHAP value).
        
        Args:
            X: Data to explain
            max_display: Number of features to display
            save_path: Path to save figure
        """
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        plt.figure(figsize=(10, 8))
        
        shap.summary_plot(
            self.shap_values,
            X,
            feature_names=self.feature_names,
            plot_type="bar",
            max_display=max_display,
            show=False
        )
        
        plt.title("SHAP Summary: Mean |SHAP| Values", 
                  fontsize=14, fontweight="bold")
        plt.xlabel("Mean |SHAP value|")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"  Saved: {save_path}")
        
        plt.close()
    
    def plot_partial_dependence(self, X_test, feature_idx, save_path=None):
        """
        Plot partial dependence for a feature.
        
        Args:
            X_test: Test data
            feature_idx: Feature index
            save_path: Path to save figure
        """
        feature_name = self.feature_names[feature_idx]
        
        # Compute partial dependence
        pd_result = partial_dependence(
            self.model, 
            X_test, 
            [feature_idx],
            percentiles=(0.05, 0.95),
            grid_resolution=50
        )
        
        pd_values = pd_result['average'][0]
        pd_grid = pd_result['grid_values'][0]
        
        plt.figure(figsize=(10, 6))
        plt.plot(pd_grid, pd_values, linewidth=2.5, color="#2ecc71")
        plt.fill_between(pd_grid, pd_values, alpha=0.3, color="#2ecc71")
        plt.xlabel(f"{feature_name} Value", fontsize=12, fontweight="bold")
        plt.ylabel("Average Model Output (Partial PD)", fontsize=12, fontweight="bold")
        plt.title(f"Partial Dependence: {feature_name}", 
                  fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"  Saved: {save_path}")
        
        plt.close()
    
    def generate_explanation_report(self, X_test, y_test=None, 
                                   predictions=None, save_dir=None):
        """
        Generate comprehensive explainability report.
        
        Args:
            X_test: Test data
            y_test: Actual labels (optional)
            predictions: Model predictions (optional)
            save_dir: Directory to save figures
        """
        if save_dir is None:
            save_dir = FIGURES_DIR
        
        print("\n[EXPLAINABILITY] Generating SHAP-based explanations...")
        
        # Compute SHAP values
        self.compute_shap_values(X_test)
        
        # 1. Summary bar plot
        print("  Creating summary bar plot...")
        self.plot_summary_bar(
            X_test, 
            max_display=15,
            save_path=os.path.join(save_dir, "16_shap_summary_bar.png")
        )
        
        # 2. Beeswarm plot
        print("  Creating beeswarm plot...")
        self.plot_beeswarm(
            X_test,
            max_display=15,
            save_path=os.path.join(save_dir, "17_shap_beeswarm.png")
        )
        
        # 3. Force diagram for highest risk loan
        if predictions is not None:
            high_risk_idx = np.argmax(predictions)
            print("  Creating force diagram for highest risk loan...")
            self.plot_force_diagram(
                X_test,
                sample_idx=high_risk_idx,
                save_path=os.path.join(save_dir, "18_shap_force_high_risk.png")
            )
        
        # 4. Partial dependence plots for top 3 features
        print("  Creating partial dependence plots...")
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        top_features = np.argsort(mean_abs_shap)[-3:][::-1]
        
        for i, feature_idx in enumerate(top_features):
            self.plot_partial_dependence(
                X_test,
                feature_idx,
                save_path=os.path.join(save_dir, f"19_partial_dependence_{i+1}.png")
            )
        
        print("  ✓ Explainability report generated")
        
    def get_feature_importance_df(self):
        """
        Get feature importance as DataFrame.
        
        Returns:
            DataFrame with features and importance scores
        """
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Mean_Abs_SHAP': mean_abs_shap
        }).sort_values('Mean_Abs_SHAP', ascending=False)
        
        return importance_df.reset_index(drop=True)
