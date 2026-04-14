"""
Configuration for Credit Risk Modeling Project
Supports environment variables and YAML config files
"""
import os
import json

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
N_SAMPLES = int(os.getenv('N_SAMPLES', 50_000))
DEFAULT_RATE = float(os.getenv('DEFAULT_RATE', 0.08))  # 8% default rate
RANDOM_SEED = int(os.getenv('RANDOM_SEED', 42))

# Model parameters
TEST_SIZE = float(os.getenv('TEST_SIZE', 0.25))
VALIDATION_SIZE = float(os.getenv('VALIDATION_SIZE', 0.15))
N_FOLDS = int(os.getenv('N_FOLDS', 5))

# Flask configuration
FLASK_HOST = os.getenv('FLASK_HOST', '0.0.0.0')
FLASK_PORT = int(os.getenv('FLASK_PORT', 5000))
FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'

# Decision engine parameters
APPROVAL_EL_THRESHOLD = float(os.getenv('APPROVAL_EL_THRESHOLD', 0.05))
BASE_INTEREST_RATE = float(os.getenv('BASE_INTEREST_RATE', 0.08))
PRICING_SCALAR = float(os.getenv('PRICING_SCALAR', 10.0))

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

# Real data configuration (for LendingClub, Kaggle, etc.)
USE_REAL_DATA = os.getenv('USE_REAL_DATA', 'False').lower() == 'true'
REAL_DATA_PATH = os.getenv('REAL_DATA_PATH', None)
DATASET_NAME = os.getenv('DATASET_NAME', 'synthetic')  # 'lending_club', 'kaggle_credit', 'synthetic'
