"""The committed models and outputs must match what the current code produces.

Fails when the pipeline changes without re-running
``python src/train_cluster_models.py``.
"""

import numpy as np
import pandas as pd
import pytest

from sales_forecasting import config, forecast_next_month, load_artifacts, run_training


@pytest.fixture(scope="module")
def fresh():
    if not config.RAW_DATA_PATH.exists():
        pytest.skip("raw data not available")
    return run_training()


def test_committed_forecast_is_current(fresh):
    committed = pd.read_csv(
        config.PROCESSED_DIR / config.FORECAST_FILE, parse_dates=["date", "forecast_month"]
    )
    current = forecast_next_month(fresh)
    assert list(committed.columns) == list(current.columns)
    assert list(committed["store_id"]) == list(current["store_id"])
    for col in ("forecast_sales", "forecast_lower", "forecast_upper", "last_month_sales"):
        np.testing.assert_allclose(committed[col], current[col], rtol=1e-6)


def test_committed_models_reproduce_forecast():
    loaded = load_artifacts()
    committed = pd.read_csv(config.PROCESSED_DIR / config.FORECAST_FILE)
    np.testing.assert_allclose(
        forecast_next_month(loaded)["forecast_sales"], committed["forecast_sales"], rtol=1e-6
    )


def test_committed_metrics_are_current(fresh):
    committed = load_artifacts().metadata["metrics"]["overall"]
    current = fresh.metadata["metrics"]["overall"]
    assert committed.keys() == current.keys()
    for method, stats in current.items():
        assert committed[method]["mae"] == pytest.approx(stats["mae"], rel=1e-6), method
