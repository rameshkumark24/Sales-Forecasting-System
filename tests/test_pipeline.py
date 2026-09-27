import json

import numpy as np
import pandas as pd
import pytest

from sales_forecasting import (
    IncompatibleModelsError,
    config,
    forecast_next_month,
    load_artifacts,
    run_training,
    save_artifacts,
)
from sales_forecasting.evaluation import BASELINES, portfolio_mape, regression_metrics


@pytest.fixture
def artifacts(raw_csv):
    return run_training(raw_csv, n_clusters=2, valid_months=3)


def test_training_reports_models_and_baselines(artifacts):
    overall = artifacts.metadata["metrics"]["overall"]
    assert {"cluster_models", "global_model", *BASELINES} <= set(overall)
    assert artifacts.metadata["selected_method"] in {"cluster_models", "global_model"}
    assert artifacts.metadata["data"]["excluded_incomplete_month"] == "2017-06-30"
    assert artifacts.metadata["validation"]["end"] == "2017-05-31"
    # Clusters come from the training window only.
    assert artifacts.metadata["validation"]["train_end"] == "2017-02-28"


def test_forecast_is_for_the_month_after_history(artifacts):
    forecast = forecast_next_month(artifacts)
    stores = artifacts.monthly["store_id"].nunique()
    assert len(forecast) == stores
    assert forecast["store_id"].is_unique
    assert (forecast["date"] == pd.Timestamp("2017-05-31")).all()
    assert (forecast["forecast_month"] == pd.Timestamp("2017-06-30")).all()
    assert (forecast["forecast_sales"] >= 0).all()
    assert (forecast["forecast_lower"] <= forecast["forecast_sales"]).all()
    assert (forecast["forecast_sales"] <= forecast["forecast_upper"]).all()


def test_last_month_sales_match_history(artifacts):
    forecast = forecast_next_month(artifacts).set_index("store_id")
    last = artifacts.monthly[artifacts.monthly["date"] == "2017-05-31"].set_index("store_id")
    pd.testing.assert_series_equal(
        forecast["last_month_sales"].sort_index(), last["sales"].sort_index(), check_names=False
    )


@pytest.mark.parametrize(
    ("strategy", "method"), [("cluster", "cluster_models"), ("global", "global_model")]
)
def test_fixed_strategy(raw_csv, strategy, method):
    arts = run_training(raw_csv, n_clusters=2, valid_months=3, strategy=strategy)
    assert arts.metadata["selected_method"] == method
    assert len(forecast_next_month(arts)) > 0


def test_rf_model_kind(raw_csv):
    arts = run_training(raw_csv, model_kind="rf", n_clusters=2, valid_months=3)
    assert arts.metadata["model_kind"] == "rf"


def test_invalid_options(raw_csv):
    with pytest.raises(ValueError):
        run_training(raw_csv, model_kind="xgb")
    with pytest.raises(ValueError):
        run_training(raw_csv, strategy="best")


def test_save_and_load_round_trip(artifacts, tmp_path):
    save_artifacts(artifacts, tmp_path / "processed", tmp_path / "models")
    for name in (config.MONTHLY_SALES_FILE, config.STORES_FILE, config.FEATURES_FILE):
        assert (tmp_path / "processed" / name).exists()

    loaded = load_artifacts(tmp_path / "processed", tmp_path / "models")
    pd.testing.assert_frame_equal(
        forecast_next_month(loaded), forecast_next_month(artifacts), check_dtype=False
    )


def test_version_mismatch_is_refused(artifacts, tmp_path):
    save_artifacts(artifacts, tmp_path / "processed", tmp_path / "models")
    meta_path = tmp_path / "models" / config.METADATA_FILE
    meta = json.loads(meta_path.read_text())
    meta["sklearn_version"] = "0.0.1"
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(IncompatibleModelsError):
        load_artifacts(tmp_path / "processed", tmp_path / "models")


def test_numpy_major_mismatch_is_refused(artifacts, tmp_path):
    save_artifacts(artifacts, tmp_path / "processed", tmp_path / "models")
    meta_path = tmp_path / "models" / config.METADATA_FILE
    meta = json.loads(meta_path.read_text())
    meta["numpy_version"] = "0.1.0"
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(IncompatibleModelsError, match="numpy"):
        load_artifacts(tmp_path / "processed", tmp_path / "models")


def test_corrupt_bundle_is_refused(artifacts, tmp_path):
    save_artifacts(artifacts, tmp_path / "processed", tmp_path / "models")
    (tmp_path / "models" / config.MODEL_BUNDLE_FILE).write_bytes(b"not a pickle")
    with pytest.raises(IncompatibleModelsError):
        load_artifacts(tmp_path / "processed", tmp_path / "models")


def test_missing_models_raise(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_artifacts(tmp_path, tmp_path)


def test_metrics():
    stats = regression_metrics([10, 20, 30], [12, 18, 33])
    assert stats["mae"] == pytest.approx(7 / 3)
    assert stats["rmse"] == pytest.approx(np.sqrt((4 + 4 + 9) / 3))
    assert stats["wape"] == pytest.approx(7 / 60)
    assert stats["bias"] == pytest.approx(1.0)
    dates = ["a", "a", "b"]
    assert portfolio_mape(dates, [10, 20, 30], [12, 18, 33]) == pytest.approx(0.05)
