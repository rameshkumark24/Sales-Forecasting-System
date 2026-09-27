"""Hold-out split, error metrics and simple benchmark forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Benchmarks every model must be compared against. Each maps to a feature
# column that already holds the benchmark's forecast for that row.
BASELINES = {
    "naive_last_month": "lag_1",
    "seasonal_naive": "lag_12",
    "moving_avg_3": "rolling_3_mean",
    "moving_avg_12": "rolling_12_mean",
    "historical_mean": "expanding_mean",
}

METHOD_NAMES = {
    "cluster_models": "Cluster-wise ML models",
    "global_model": "Single global ML model",
    "naive_last_month": "Naive (last month)",
    "seasonal_naive": "Seasonal naive (same month last year)",
    "moving_avg_3": "3-month moving average",
    "moving_avg_12": "12-month moving average",
    "historical_mean": "Store historical mean",
}


def time_split(
    frame: pd.DataFrame, valid_months: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split on time: the last ``valid_months`` months form the validation set."""
    months = np.sort(frame["date"].unique())
    if len(months) <= valid_months:
        raise ValueError(
            f"Need more than {valid_months} months of feature rows, got {len(months)}."
        )
    cutoff = pd.Timestamp(months[-valid_months - 1])
    train = frame[frame["date"] <= cutoff]
    valid = frame[frame["date"] > cutoff]
    return train, valid, cutoff


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """MAE, RMSE, WAPE (sum |error| / sum |actual|) and bias (mean error)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_pred - y_true
    total = np.abs(y_true).sum()
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "wape": float(np.abs(error).sum() / total) if total else float("nan"),
        "bias": float(np.mean(error)),
        "n": len(y_true),
    }


def portfolio_mape(dates, y_true, y_pred) -> float:
    """Mean absolute % error of the all-store total, month by month.

    Store-level MAE can hide a model that is systematically low; this checks
    the headline number a planner would actually use.
    """
    totals = pd.DataFrame({"date": dates, "y": y_true, "p": y_pred}).groupby("date").sum()
    totals = totals[totals["y"] != 0]
    return float(((totals["p"] - totals["y"]).abs() / totals["y"]).mean())


def baseline_predictions(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {name: frame[column] for name, column in BASELINES.items()}


def improvement_pct(model_mae: float, reference_mae: float) -> float:
    """Percentage MAE reduction of a model versus a reference (positive = better)."""
    return float((reference_mae - model_mae) / reference_mae * 100) if reference_mae else 0.0
