"""Feature engineering shared by training and forecasting.

Every feature for month ``t`` is computed only from sales of months before
``t``. Training rows and next-month forecast rows go through the same
function, so the model always sees features built the same way.
"""

from __future__ import annotations

import pandas as pd

LAGS = (1, 2, 3, 6, 12)
ROLLING_WINDOWS = (3, 6, 12)
CALENDAR_FEATURES = ["month", "quarter", "year"]

FEATURE_COLUMNS = [
    *(f"lag_{lag}" for lag in LAGS),
    *(f"rolling_{w}_mean" for w in ROLLING_WINDOWS),
    "rolling_6_std",
    "expanding_mean",
    *CALENDAR_FEATURES,
]

# A row needs this many prior months before every feature is defined.
MIN_HISTORY_MONTHS = max(max(LAGS), max(ROLLING_WINDOWS))


def add_features(monthly: pd.DataFrame) -> pd.DataFrame:
    """Add lag, rolling and calendar features to a monthly sales frame.

    ``monthly`` must hold ``date``, ``store_id`` and ``sales`` with one row
    per store and consecutive calendar month (see ``aggregate_monthly``).
    Rows without enough history keep NaN features.
    """
    df = monthly.sort_values(["store_id", "date"], ignore_index=True)
    by_store = df.groupby("store_id", sort=False)["sales"]

    for lag in LAGS:
        df[f"lag_{lag}"] = by_store.shift(lag)

    # Sales strictly before the current month, still grouped per store so
    # rolling windows never cross from one store into the next.
    past = by_store.shift(1).groupby(df["store_id"], sort=False)
    for window in ROLLING_WINDOWS:
        df[f"rolling_{window}_mean"] = past.transform(lambda s, w=window: s.rolling(w).mean())
    df["rolling_6_std"] = past.transform(lambda s: s.rolling(6).std())
    df["expanding_mean"] = past.transform(lambda s: s.expanding().mean())

    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["year"] = df["date"].dt.year
    return df


def build_training_frame(monthly: pd.DataFrame) -> pd.DataFrame:
    """Feature rows with a known target and complete history."""
    return add_features(monthly).dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)


def build_next_month_features(monthly: pd.DataFrame) -> pd.DataFrame:
    """Feature rows for the month right after each store's last observed month.

    A placeholder row with unknown sales is appended per store and run through
    ``add_features``; since features only look backwards, the placeholder's
    own (missing) sales never leak into its features.
    """
    last = monthly.sort_values("date").groupby("store_id", as_index=False).tail(1)
    placeholder = pd.DataFrame(
        {
            "date": last["date"] + pd.offsets.MonthEnd(1),
            "store_id": last["store_id"],
            "sales": float("nan"),
        }
    )
    extended = pd.concat([monthly[["date", "store_id", "sales"]], placeholder], ignore_index=True)
    featured = add_features(extended)

    is_future = featured.set_index(["store_id", "date"]).index.isin(
        placeholder.set_index(["store_id", "date"]).index
    )
    return featured[is_future].drop(columns="sales").reset_index(drop=True)
