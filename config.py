"""
Configuration for Credit Risk Modeling Project
"""
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
DATA_DIR = os.path.join(OUTPUT_DIR, "data")
DOCS_DIR = os.path.join(BASE_DIR, "docs")

for d in [OUTPUT_DIR, FIGURES_DIR, MODELS_DIR, DATA_DIR, DOCS_DIR]:
    os.makedirs(d, exist_ok=True)

# Data generation
N_SAMPLES = 50_000
DEFAULT_RATE = 0.08  # 8% default rate
RANDOM_SEED = 42

# Model parameters
TEST_SIZE = 0.25
VALIDATION_SIZE = 0.15
N_FOLDS = 5

# Stress testing scenarios
STRESS_SCENARIOS = {
    "baseline": {"gdp_shock": 0.0, "unemployment_shock": 0.0, "rate_shock": 0.0},
    "mild_recession": {"gdp_shock": -0.02, "unemployment_shock": 0.03, "rate_shock": 0.01},
    "severe_recession": {"gdp_shock": -0.05, "unemployment_shock": 0.07, "rate_shock": 0.025},
    "deep_depression": {"gdp_shock": -0.10, "unemployment_shock": 0.12, "rate_shock": 0.04},
}

# Basel III parameters
CONFIDENCE_LEVEL = 0.999
MATURITY = 2.5
LGD_DOWNTURN_MULTIPLIER = 1.2
