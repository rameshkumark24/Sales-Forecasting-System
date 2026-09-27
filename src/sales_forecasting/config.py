"""Project-wide paths and defaults.

Paths are resolved from this file's location, so scripts work no matter which
directory they are launched from.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "50000 Sales Records.csv"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

MONTHLY_SALES_FILE = "monthly_sales.csv"
FEATURES_FILE = "features.csv"
STORES_FILE = "stores.csv"
FORECAST_FILE = "next_month_forecast_cluster.csv"
MODEL_BUNDLE_FILE = "forecast_models.joblib"
METADATA_FILE = "metadata.json"

# Modelling defaults
N_CLUSTERS = 3
VALIDATION_MONTHS = 6
RANDOM_STATE = 42
DEFAULT_MODEL_KIND = "hgb"
# Central 80% prediction interval, estimated from hold-out residuals.
INTERVAL_QUANTILES = (0.10, 0.90)
