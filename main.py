import sys
import os
import time
import json
import warnings

warnings.filterwarnings("ignore")

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    N_SAMPLES, DEFAULT_RATE, RANDOM_SEED, TEST_SIZE,
    OUTPUT_DIR, FIGURES_DIR, MODELS_DIR, DATA_DIR, DOCS_DIR,
)
from src.data_generator import generate_credit_data, add_derived_features
from src.eda import run_eda
from src.feature_engineering import FeatureEngineer
from src.pd_model import PDModelSuite
from src.lgd_model import LGDModel
from src.ead_model import EADModel
from src.scorecard import CreditScorecard
from src.validation import full_validation, plot_decile_analysis
from src.stress_testing import StressTester
from src.capital import BaselCapitalCalculator


def main():
    start_time = time.time()
    print("=" * 70)
    print("  CREDIT RISK MODELING — END-TO-END PIPELINE")
    print("=" * 70)

    # =========================================================================
    # STEP 1: Data Generation
    # =========================================================================
    print("\n[1/10] Generating synthetic credit portfolio data...")
    df = generate_credit_data(n_samples=N_SAMPLES, default_rate=DEFAULT_RATE, seed=RANDOM_SEED)
    df = add_derived_features(df)
    df.to_csv(os.path.join(DATA_DIR, "credit_portfolio.csv"), index=False)
    print(f"  Dataset: {len(df):,} loans | Default rate: {df['default'].mean():.2%}")

    # =========================================================================
    # STEP 2: Exploratory Data Analysis
    # =========================================================================
    print("\n[2/10] Running Exploratory Data Analysis...")
    eda_summary = run_eda(df)

    # =========================================================================
    # STEP 3: Train/Test Split & Feature Engineering
    # =========================================================================
    print("\n[3/10] Feature Engineering & Train/Test Split...")
    df_train, df_test = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=df["default"]
    )
    print(f"  Train: {len(df_train):,} | Test: {len(df_test):,}")

    fe = FeatureEngineer()
    X_train, y_train = fe.fit_transform(df_train)
    X_test, y_test = fe.transform(df_test)
    feature_names = fe.get_feature_importance_names()
    print(f"  Features: {len(feature_names)}")

    # =========================================================================
    # STEP 4: PD Model Training
    # =========================================================================
    print("\n[4/10] Training PD Models...")
    pd_suite = PDModelSuite()
    cv_results = pd_suite.train_all(X_train, y_train)
    pd_suite.plot_model_comparison()
    pd_suite.plot_feature_importance(feature_names)
    pd_suite.save_results()

    # Predictions
    y_train_pred = pd_suite.predict_pd(X_train)
    y_test_pred = pd_suite.predict_pd(X_test)

    # =========================================================================
    # STEP 5: Model Validation
    # =========================================================================
    print("\n[5/10] Running Model Validation...")
    validation_results = full_validation(
        y_train, y_train_pred, y_test, y_test_pred,
        model_name=pd_suite.best_model_name,
    )
    decile_stats = plot_decile_analysis(y_test, y_test_pred)

    print(f"  AUC (Train): {validation_results['auc_train']:.4f}")
    print(f"  AUC (Test):  {validation_results['auc_test']:.4f}")
    print(f"  KS (Test):   {validation_results['ks_test']:.4f}")
    print(f"  Gini (Test): {validation_results['gini_test']:.4f}")
    print(f"  Brier Score: {validation_results['brier_score']:.4f}")
    print(f"  PSI:         {validation_results['psi']:.4f}")

    # =========================================================================
    # STEP 6: Credit Scorecard
    # =========================================================================
    print("\n[6/10] Building Credit Scorecard...")
    scorecard = CreditScorecard()
    scorecard_features = [
        "fico_score", "credit_utilization", "dti_ratio", "annual_income",
        "loan_amount", "interest_rate", "employment_length", "delinq_2yrs",
        "inquiries_6mo", "revolving_balance", "home_ownership", "loan_purpose",
    ]
    iv_results = scorecard.fit(df_train, scorecard_features)

    scores = scorecard.compute_score(y_test_pred)
    scorecard.plot_scorecard_results(scores, np.array(y_test), y_test_pred)

    # =========================================================================
    # STEP 7: LGD Modeling
    # =========================================================================
    print("\n[7/10] Training LGD Model...")
    lgd_model = LGDModel()

    lgd_features = [c for c in feature_names if c in df_train.columns]
    if not lgd_features:
        lgd_features = [c for c in df_train.select_dtypes(include=[np.number]).columns
                        if c not in ["default", "lgd", "ead", "credit_limit", "drawn_amount"]]

    X_lgd_train, y_lgd_train, X_lgd_test, y_lgd_test = lgd_model.prepare_lgd_data(
        df_train, df_test, lgd_features
    )

    if len(X_lgd_train) > 0:
        lgd_model.train(X_lgd_train, y_lgd_train)
        lgd_metrics = lgd_model.evaluate(X_lgd_test, y_lgd_test)
        y_lgd_pred = lgd_model.predict(X_lgd_test)
        lgd_model.plot_lgd_analysis(y_lgd_test, y_lgd_pred)

    # =========================================================================
    # STEP 8: EAD Modeling
    # =========================================================================
    print("\n[8/10] Training EAD Model...")
    ead_model = EADModel()
    X_ead_train, y_ead_train, X_ead_test, y_ead_test = ead_model.prepare_ead_data(
        df_train, df_test, lgd_features
    )

    if len(X_ead_train) > 0:
        ead_model.train(X_ead_train, y_ead_train)
        ead_metrics = ead_model.evaluate(X_ead_test, y_ead_test)
        y_ead_pred = ead_model.predict_ccf(X_ead_test)
        ead_model.plot_ead_analysis(y_ead_test, y_ead_pred)

    # =========================================================================
    # STEP 9: Stress Testing
    # =========================================================================
    print("\n[9/10] Running Stress Tests...")
    stress_tester = StressTester(pd_suite, lgd_model, fe)
    stress_results = stress_tester.run_stress_tests(X_test, df_test)
    stress_tester.plot_stress_results()

    # =========================================================================
    # STEP 10: Basel III Capital Calculation
    # =========================================================================
    print("\n[10/10] Computing Basel III Regulatory Capital...")
    capital_calc = BaselCapitalCalculator()

    pd_portfolio = y_test_pred
    lgd_portfolio = np.full_like(pd_portfolio, 0.40)  # Supervisory LGD
    ead_portfolio = df_test["loan_amount"].values

    capital_results = capital_calc.compute_portfolio_capital(
        pd_portfolio, lgd_portfolio, ead_portfolio
    )
    capital_calc.plot_capital_analysis(pd_portfolio, lgd_portfolio, ead_portfolio)

    # =========================================================================
    # Save all results
    # =========================================================================
    all_results = {
        "eda_summary": {k: round(v, 4) if isinstance(v, float) else v
                        for k, v in eda_summary.items()},
        "validation": {k: round(v, 4) for k, v in validation_results.items()},
        "capital": {
            "total_ead": round(capital_results["total_ead"], 2),
            "total_rwa": round(capital_results["total_rwa"], 2),
            "rwa_density": round(capital_results["rwa_density"], 4),
            "total_el": round(capital_results["total_el"], 2),
            "el_ratio": round(capital_results["el_ratio"], 6),
            "total_capital": round(capital_results["total_capital"], 2),
            "capital_ratio": round(capital_results["capital_ratio"], 4),
        },
        "stress_testing": {
            scenario: {
                "avg_pd": round(res["avg_pd"], 4),
                "el_ratio": round(res["el_ratio"], 6),
                "portfolio_el": round(res["portfolio_el"], 2),
            }
            for scenario, res in stress_results.items()
        },
    }

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(os.path.join(OUTPUT_DIR, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"  PIPELINE COMPLETE — {elapsed:.1f}s")
    print(f"  Figures saved to: {FIGURES_DIR}")
    print(f"  Results saved to: {OUTPUT_DIR}")
    print(f"{'=' * 70}")

    # Generate thesis PDF
    print("\n  Generating thesis PDF...")
    try:
        from generate_thesis_pdf import generate_thesis
        generate_thesis(all_results)
        print(f"  Thesis saved to: {DOCS_DIR}")
    except Exception as e:
        print(f"  PDF generation error: {e}")
        print("  Run 'python generate_thesis_pdf.py' separately after installing reportlab")

    return all_results


if __name__ == "__main__":
    results = main()
