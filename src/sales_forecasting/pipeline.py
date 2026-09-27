"""End-to-end training and next-month forecasting."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .clustering import cluster_labels, fit_store_clusters, store_profile
from .data import (
    aggregate_monthly,
    incomplete_last_month,
    load_raw_orders,
    store_regions,
)
from .evaluation import (
    BASELINES,
    baseline_predictions,
    improvement_pct,
    portfolio_mape,
    regression_metrics,
    time_split,
)
from .features import FEATURE_COLUMNS, build_next_month_features, build_training_frame
from .modeling import (
    MODEL_KINDS,
    load_metadata,
    load_models,
    make_model,
    runtime_versions,
    save_models,
    version_mismatch,
)

logger = logging.getLogger(__name__)

GLOBAL = "global"
MIN_TRAIN_ROWS = 50
# Forecasting strategies that can be chosen, keyed by CLI name -> metrics key.
STRATEGIES = {"cluster": "cluster_models", "global": "global_model"}


class IncompatibleModelsError(RuntimeError):
    """Saved models can't be loaded safely here (library versions differ or the file is bad)."""


@dataclass
class Artifacts:
    """Everything needed to forecast: history, store tiers, models and metadata."""

    monthly: pd.DataFrame
    stores: pd.DataFrame
    models: dict
    metadata: dict
    features: pd.DataFrame | None = None


def _fit(frame: pd.DataFrame, kind: str, random_state: int):
    model = make_model(kind, random_state)
    model.fit(frame[FEATURE_COLUMNS], frame["sales"])
    return model


def _predict(model, frame: pd.DataFrame) -> np.ndarray:
    return np.clip(model.predict(frame[FEATURE_COLUMNS]), 0.0, None)


def _fit_cluster_models(frame, clusters, kind, random_state) -> dict:
    """One model per cluster; clusters with too little data fall back to the global model."""
    models = {}
    for cl in clusters:
        subset = frame[frame["cluster"] == cl]
        if len(subset) < MIN_TRAIN_ROWS:
            logger.warning("Cluster %s has only %d rows; using the global model.", cl, len(subset))
            continue
        models[int(cl)] = _fit(subset, kind, random_state)
    return models


def _predict_with(models: dict, frame: pd.DataFrame, method: str) -> pd.Series:
    """Predict with the global model, or with each row's cluster model."""
    if method == "global_model":
        return pd.Series(_predict(models[GLOBAL], frame), index=frame.index)
    preds = pd.Series(np.nan, index=frame.index)
    for cl, part in frame.groupby("cluster"):
        model = models["clusters"].get(int(cl), models[GLOBAL])
        preds.loc[part.index] = _predict(model, part)
    return preds


def _metric_block(frame: pd.DataFrame, predictions: dict[str, pd.Series]) -> dict:
    y_true = frame["sales"]
    block = {}
    for name, pred in predictions.items():
        pred = pred.loc[frame.index]
        block[name] = regression_metrics(y_true, pred)
        block[name]["portfolio_mape"] = portfolio_mape(frame["date"], y_true, pred)
    naive = block["naive_last_month"]["mae"]
    best_baseline = min(block[b]["mae"] for b in BASELINES)
    for stats in block.values():
        stats["improvement_vs_naive_pct"] = improvement_pct(stats["mae"], naive)
        stats["improvement_vs_best_baseline_pct"] = improvement_pct(stats["mae"], best_baseline)
    return block


def run_training(
    raw_path: Path = config.RAW_DATA_PATH,
    *,
    model_kind: str = config.DEFAULT_MODEL_KIND,
    n_clusters: int = config.N_CLUSTERS,
    valid_months: int = config.VALIDATION_MONTHS,
    drop_incomplete_last_month: bool = True,
    strategy: str = "auto",
    random_state: int = config.RANDOM_STATE,
) -> Artifacts:
    """Build features, evaluate on a hold-out period, then refit on all data.

    1. Aggregate raw orders into a gap-free monthly series per store.
    2. Hold out the last ``valid_months`` months.
    3. Cluster stores using the training period only (no look-ahead).
    4. Score cluster-wise models, a single global model and naive baselines
       on the hold-out months. ``strategy="auto"`` forecasts with whichever
       of the two ML approaches has the lower hold-out MAE.
    5. Refit the models on every month so the forecast uses all history.
    """
    if model_kind not in MODEL_KINDS:
        raise ValueError(f"Unknown model kind {model_kind!r}; choose from {sorted(MODEL_KINDS)}.")
    if strategy != "auto" and strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}; choose auto, cluster or global.")

    orders = load_raw_orders(raw_path)
    partial = incomplete_last_month(orders) if drop_incomplete_last_month else None
    monthly = aggregate_monthly(orders, drop_incomplete_last_month=drop_incomplete_last_month)
    frame = build_training_frame(monthly)
    _, _, cutoff = time_split(frame, valid_months)

    profile = store_profile(monthly[monthly["date"] <= cutoff])
    stores = fit_store_clusters(profile, n_clusters, random_state)
    stores = store_regions(orders).merge(stores, on="store_id", how="inner")
    cluster_of = stores.set_index("store_id")["cluster"]

    frame = frame.assign(cluster=frame["store_id"].map(cluster_of))
    frame = frame.dropna(subset=["cluster"]).astype({"cluster": int})
    train = frame[frame["date"] <= cutoff]
    valid = frame[frame["date"] > cutoff]
    clusters = sorted(frame["cluster"].unique())

    # --- Hold-out evaluation -------------------------------------------------
    eval_models = {
        "clusters": _fit_cluster_models(train, clusters, model_kind, random_state),
        GLOBAL: _fit(train, model_kind, random_state),
    }
    predictions = {
        method: _predict_with(eval_models, valid, method) for method in STRATEGIES.values()
    }
    predictions.update(baseline_predictions(valid))
    metrics = {"overall": _metric_block(valid, predictions), "by_cluster": {}}
    for cl, part in valid.groupby("cluster"):
        metrics["by_cluster"][str(cl)] = _metric_block(part, predictions)

    if strategy == "auto":
        selected = min(STRATEGIES.values(), key=lambda m: metrics["overall"][m]["mae"])
    else:
        selected = STRATEGIES[strategy]

    # Empirical prediction intervals from hold-out residuals (actual - forecast).
    lo_q, hi_q = config.INTERVAL_QUANTILES
    residuals = valid["sales"] - predictions[selected]
    intervals = {GLOBAL: [float(residuals.quantile(lo_q)), float(residuals.quantile(hi_q))]}
    for cl, res in residuals.groupby(valid["cluster"]):
        intervals[str(cl)] = [float(res.quantile(lo_q)), float(res.quantile(hi_q))]

    # --- Final models on all available history ------------------------------
    final_models = {
        "clusters": _fit_cluster_models(frame, clusters, model_kind, random_state),
        GLOBAL: _fit(frame, model_kind, random_state),
    }

    metadata = {
        "model_kind": model_kind,
        "model_description": MODEL_KINDS[model_kind],
        "selected_method": selected,
        "selection": "lowest hold-out MAE" if strategy == "auto" else "fixed by user",
        **runtime_versions(),
        "feature_columns": FEATURE_COLUMNS,
        "n_clusters": n_clusters,
        "cluster_labels": {str(k): v for k, v in cluster_labels(n_clusters).items()},
        "cluster_store_counts": {
            str(k): int(v) for k, v in stores["cluster"].value_counts().sort_index().items()
        },
        "data": {
            "first_month": monthly["date"].min().strftime("%Y-%m-%d"),
            "last_month": monthly["date"].max().strftime("%Y-%m-%d"),
            "stores": int(monthly["store_id"].nunique()),
            "excluded_incomplete_month": partial.strftime("%Y-%m-%d") if partial else None,
            "raw_last_order_date": orders["order_date"].max().strftime("%Y-%m-%d"),
        },
        "validation": {
            "months": valid_months,
            "train_end": cutoff.strftime("%Y-%m-%d"),
            "start": valid["date"].min().strftime("%Y-%m-%d"),
            "end": valid["date"].max().strftime("%Y-%m-%d"),
            "rows": len(valid),
        },
        "interval": {
            "coverage": round(hi_q - lo_q, 2),
            "residual_quantiles": intervals,
        },
        "metrics": metrics,
    }
    return Artifacts(
        monthly=monthly,
        stores=stores,
        models=final_models,
        metadata=metadata,
        features=frame,
    )


def forecast_next_month(artifacts: Artifacts) -> pd.DataFrame:
    """Forecast the first month after the history for every clustered store."""
    monthly, stores, models = artifacts.monthly, artifacts.stores, artifacts.models
    intervals = artifacts.metadata["interval"]["residual_quantiles"]

    rows = build_next_month_features(monthly)
    rows = rows.merge(stores[["store_id", "region", "cluster", "cluster_label"]], on="store_id")
    missing_history = rows[FEATURE_COLUMNS].isna().any(axis=1)
    if missing_history.any():
        logger.warning(
            "Skipping %d store(s) with under a year of history: %s",
            int(missing_history.sum()),
            ", ".join(rows.loc[missing_history, "store_id"]),
        )
        rows = rows[~missing_history]
    if rows.empty:
        raise ValueError("No store has enough history to forecast.")

    forecast = _predict_with(models, rows, artifacts.metadata["selected_method"])
    lower = rows["cluster"].map(lambda c: intervals.get(str(c), intervals[GLOBAL])[0])
    upper = rows["cluster"].map(lambda c: intervals.get(str(c), intervals[GLOBAL])[1])

    last = monthly.sort_values("date").groupby("store_id").tail(1)
    last = last.rename(columns={"sales": "last_month_sales"})
    out = rows[["store_id", "region", "cluster", "cluster_label", "date"]].rename(
        columns={"date": "forecast_month"}
    )
    out = out.merge(last, on="store_id")
    out["forecast_sales"] = forecast.to_numpy()
    out["forecast_lower"] = np.clip(forecast.to_numpy() + lower.to_numpy(), 0.0, None)
    out["forecast_upper"] = np.clip(forecast.to_numpy() + upper.to_numpy(), 0.0, None)
    out["change_pct"] = np.where(
        out["last_month_sales"] > 0,
        (out["forecast_sales"] / out["last_month_sales"] - 1) * 100,
        np.nan,
    )
    columns = [
        "store_id",
        "region",
        "cluster",
        "cluster_label",
        "date",
        "last_month_sales",
        "forecast_month",
        "forecast_sales",
        "forecast_lower",
        "forecast_upper",
        "change_pct",
    ]
    return out[columns].sort_values("store_id", ignore_index=True)


def save_artifacts(
    artifacts: Artifacts,
    processed_dir: Path = config.PROCESSED_DIR,
    models_dir: Path = config.MODELS_DIR,
) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    region_of = artifacts.stores.set_index("store_id")["region"]
    monthly = artifacts.monthly.assign(region=artifacts.monthly["store_id"].map(region_of))
    monthly[["date", "store_id", "region", "sales"]].to_csv(
        processed_dir / config.MONTHLY_SALES_FILE, index=False, date_format="%Y-%m-%d"
    )
    artifacts.stores.to_csv(processed_dir / config.STORES_FILE, index=False)
    if artifacts.features is not None:
        cols = ["date", "store_id", "cluster", "sales", *FEATURE_COLUMNS]
        artifacts.features[cols].to_csv(
            processed_dir / config.FEATURES_FILE, index=False, date_format="%Y-%m-%d"
        )
    save_models(
        artifacts.models,
        artifacts.metadata,
        models_dir,
        config.MODEL_BUNDLE_FILE,
        config.METADATA_FILE,
    )


def load_artifacts(
    processed_dir: Path = config.PROCESSED_DIR,
    models_dir: Path = config.MODELS_DIR,
) -> Artifacts:
    """Load saved history, store tiers, models and metadata from disk."""
    monthly_path = processed_dir / config.MONTHLY_SALES_FILE
    stores_path = processed_dir / config.STORES_FILE
    for path in (monthly_path, stores_path):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run `python src/train_cluster_models.py`.")

    metadata = load_metadata(models_dir / config.METADATA_FILE)
    # Check before unpickling: a model pickled by another scikit-learn version
    # can fail to load or silently predict wrong values.
    problem = version_mismatch(metadata)
    if problem:
        raise IncompatibleModelsError(
            f"{problem} Retrain with `python src/train_cluster_models.py`."
        )
    monthly = pd.read_csv(monthly_path, parse_dates=["date"])
    stores = pd.read_csv(stores_path)
    try:
        models = load_models(models_dir / config.MODEL_BUNDLE_FILE)
    except FileNotFoundError:
        raise
    except Exception as exc:  # any unpickling failure means the bundle is unusable here
        raise IncompatibleModelsError(
            f"Could not load saved models ({exc}). "
            "Retrain with `python src/train_cluster_models.py`."
        ) from exc
    return Artifacts(
        monthly=monthly[["date", "store_id", "sales"]],
        stores=stores,
        models=models,
        metadata=metadata,
    )
