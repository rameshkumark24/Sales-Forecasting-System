"""Loading raw orders and turning them into a clean monthly sales series."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

RAW_COLUMNS = {
    "Order Date": "order_date",
    "Country": "store_id",
    "Region": "region",
    "Total Revenue": "revenue",
}


def load_raw_orders(path: str | Path) -> pd.DataFrame:
    """Read the raw order export and keep the columns the pipeline needs.

    Country and region names are stripped because the export contains
    trailing spaces (e.g. ``"Moldova "``).
    """
    orders = pd.read_csv(path, usecols=list(RAW_COLUMNS))
    orders = orders.rename(columns=RAW_COLUMNS)
    orders["order_date"] = pd.to_datetime(orders["order_date"], format="%m/%d/%Y")
    orders["store_id"] = orders["store_id"].astype(str).str.strip()
    orders["region"] = orders["region"].astype(str).str.strip()
    orders["revenue"] = pd.to_numeric(orders["revenue"], errors="raise")
    return orders


def incomplete_last_month(orders: pd.DataFrame) -> pd.Timestamp | None:
    """Return the month-end date of the final month if it is only partly covered.

    A month counts as incomplete when the latest order falls before the last
    calendar day of that month (the raw file stops on 2017-07-28).
    """
    last_order = orders["order_date"].max().normalize()
    month_end = last_order + pd.offsets.MonthEnd(0)
    return month_end if last_order < month_end else None


def aggregate_monthly(
    orders: pd.DataFrame, drop_incomplete_last_month: bool = True
) -> pd.DataFrame:
    """Aggregate orders to one row per store and calendar month.

    Months without orders inside a store's active period are filled with zero
    sales, so that ``shift(1)`` really means "previous calendar month".
    """
    orders = orders.assign(date=orders["order_date"] + pd.offsets.MonthEnd(0))

    partial = incomplete_last_month(orders) if drop_incomplete_last_month else None
    if partial is not None:
        logger.warning(
            "Dropping %s: raw data ends on %s, so that month is incomplete.",
            partial.strftime("%Y-%m"),
            orders["order_date"].max().date(),
        )
        orders = orders[orders["date"] < partial]

    monthly = orders.groupby(["store_id", "date"], as_index=False)["revenue"].sum()
    monthly = monthly.rename(columns={"revenue": "sales"})

    last_month = monthly["date"].max()
    frames = []
    for store_id, group in monthly.groupby("store_id", sort=True):
        calendar = pd.date_range(group["date"].min(), last_month, freq="ME")
        filled = (
            group.set_index("date")["sales"]
            .reindex(calendar, fill_value=0.0)
            .rename_axis("date")
            .reset_index()
        )
        filled["store_id"] = store_id
        frames.append(filled)

    result = pd.concat(frames, ignore_index=True)
    return result[["date", "store_id", "sales"]].sort_values(
        ["store_id", "date"], ignore_index=True
    )


def store_regions(orders: pd.DataFrame) -> pd.DataFrame:
    """One row per store with its sales region."""
    return (
        orders.groupby("store_id", as_index=False)["region"]
        .agg(lambda s: s.mode().iat[0])
        .sort_values("store_id", ignore_index=True)
    )
