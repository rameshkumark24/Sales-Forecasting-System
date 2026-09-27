import numpy as np
import pandas as pd
import pytest

from sales_forecasting.features import (
    FEATURE_COLUMNS,
    MIN_HISTORY_MONTHS,
    add_features,
    build_next_month_features,
    build_training_frame,
)


def monthly_frame(n_months: int = 30) -> pd.DataFrame:
    dates = pd.date_range("2015-01-31", periods=n_months, freq="ME")
    frames = [
        pd.DataFrame({"date": dates, "store_id": store, "sales": np.arange(n_months) + offset})
        for store, offset in (("A", 0.0), ("B", 1000.0))
    ]
    return pd.concat(frames, ignore_index=True)


def test_lags_follow_calendar_months():
    feats = add_features(monthly_frame()).set_index(["store_id", "date"])
    row = feats.loc[("A", pd.Timestamp("2016-06-30"))]
    assert row["sales"] == 17
    assert row["lag_1"] == 16
    assert row["lag_12"] == 5
    assert row["rolling_3_mean"] == pytest.approx((16 + 15 + 14) / 3)
    assert row["expanding_mean"] == pytest.approx(np.mean(np.arange(17)))
    assert (row["month"], row["quarter"], row["year"]) == (6, 2, 2016)


def test_windows_do_not_cross_stores():
    feats = add_features(monthly_frame())
    first_b = feats[feats["store_id"] == "B"].iloc[0]
    assert np.isnan(first_b["lag_1"])
    assert np.isnan(first_b["rolling_3_mean"])
    assert np.isnan(first_b["expanding_mean"])


def test_features_do_not_use_current_month():
    base = monthly_frame()
    changed = base.copy()
    changed.loc[changed["date"] == "2016-06-30", "sales"] = 1e9
    a = add_features(base).set_index(["store_id", "date"])
    b = add_features(changed).set_index(["store_id", "date"])
    key = pd.Timestamp("2016-06-30")
    pd.testing.assert_series_equal(
        a.xs(key, level="date")[FEATURE_COLUMNS].stack(),
        b.xs(key, level="date")[FEATURE_COLUMNS].stack(),
    )


def test_training_frame_drops_short_history():
    frame = build_training_frame(monthly_frame())
    assert not frame[FEATURE_COLUMNS].isna().any().any()
    assert len(frame) == 2 * (30 - MIN_HISTORY_MONTHS)


def test_next_month_rows_match_training_features():
    """Features built for the forecast equal those built once the month is known."""
    full = monthly_frame(31)
    history = full[full["date"] < full["date"].max()]

    future = build_next_month_features(history).set_index("store_id")
    known = add_features(full)
    known = known[known["date"] == full["date"].max()].set_index("store_id")

    assert (future["date"] == pd.Timestamp("2017-07-31")).all()
    pd.testing.assert_frame_equal(
        future[FEATURE_COLUMNS].sort_index(), known[FEATURE_COLUMNS].sort_index()
    )
