"""
Feature Engineering Pipeline for Credit Risk Models
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


class FeatureEngineer:
    """Builds and transforms features for credit risk models."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy="median")
        self.feature_names = None
        self.numeric_cols = None
        self.categorical_cols = None

    def fit_transform(self, df, target="default"):
        """Fit on training data and transform."""
        df = self._add_engineered_features(df)
        X, y = self._prepare_features(df, target)
        X_imputed = pd.DataFrame(
            self.imputer.fit_transform(X), columns=X.columns, index=X.index
        )
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_imputed), columns=X_imputed.columns, index=X_imputed.index
        )
        self.feature_names = list(X_scaled.columns)
        return X_scaled, y

    def transform(self, df, target="default"):
        """Transform new data using fitted parameters."""
        df = self._add_engineered_features(df)
        X, y = self._prepare_features(df, target)
        X_imputed = pd.DataFrame(
            self.imputer.transform(X), columns=X.columns, index=X.index
        )
        X_scaled = pd.DataFrame(
            self.scaler.transform(X_imputed), columns=X_imputed.columns, index=X_imputed.index
        )
        return X_scaled, y

    def _add_engineered_features(self, df):
        df = df.copy()
        df["loan_to_income"] = df["loan_amount"] / df["annual_income"]
        df["balance_to_income"] = df["revolving_balance"] / df["annual_income"]
        df["income_per_credit_line"] = df["annual_income"] / df["num_credit_lines"]
        df["high_utilization"] = (df["credit_utilization"] > 0.5).astype(int)
        df["has_delinquency"] = (df["delinq_2yrs"] > 0).astype(int)
        df["has_public_record"] = (df["public_records"] > 0).astype(int)
        df["monthly_payment_est"] = (
            df["loan_amount"]
            * (df["interest_rate"] / 100 / 12)
            / (1 - (1 + df["interest_rate"] / 100 / 12) ** (-df["term_months"]))
        )
        df["payment_to_income"] = df["monthly_payment_est"] / (df["annual_income"] / 12)
        df["real_rate"] = df["interest_rate"] / 100 - df["gdp_growth"]
        df["fico_x_utilization"] = df["fico_score"] * df["credit_utilization"]
        df["fico_x_dti"] = df["fico_score"] * df["dti_ratio"]
        df["log_income"] = np.log1p(df["annual_income"])
        df["log_loan"] = np.log1p(df["loan_amount"])
        df["log_balance"] = np.log1p(df["revolving_balance"])
        return df

    def _prepare_features(self, df, target):
        exclude_cols = [target, "lgd", "ead", "credit_limit", "drawn_amount", "fico_bucket"]
        self.categorical_cols = ["home_ownership", "loan_purpose"]
        self.numeric_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in exclude_cols
        ]

        # Encode categoricals
        dummies = pd.get_dummies(df[self.categorical_cols], drop_first=True, dtype=int)

        X = pd.concat([df[self.numeric_cols], dummies], axis=1)
        y = df[target] if target in df.columns else None
        return X, y

    def get_feature_importance_names(self):
        return self.feature_names
