import pandas as pd
import pytest

from sales_forecasting.data import (
    aggregate_monthly,
    incomplete_last_month,
    load_raw_orders,
    store_regions,
)


def test_store_names_are_stripped(raw_csv):
    orders = load_raw_orders(raw_csv)
    assert "Bravo" in set(orders["store_id"])
    assert not any(name != name.strip() for name in orders["store_id"].unique())


def test_incomplete_last_month_detected(raw_csv):
    orders = load_raw_orders(raw_csv)
    assert incomplete_last_month(orders) == pd.Timestamp("2017-06-30")


def test_complete_last_month_not_flagged(raw_csv):
    orders = load_raw_orders(raw_csv)
    orders = orders[orders["order_date"] < "2017-06-01"]
    extra = orders.iloc[[0]].assign(order_date=pd.Timestamp("2017-05-31"))
    assert incomplete_last_month(pd.concat([orders, extra])) is None


def test_partial_month_dropped_by_default(raw_csv):
    orders = load_raw_orders(raw_csv)
    assert aggregate_monthly(orders)["date"].max() == pd.Timestamp("2017-05-31")
    kept = aggregate_monthly(orders, drop_incomplete_last_month=False)
    assert kept["date"].max() == pd.Timestamp("2017-06-30")


def test_missing_months_are_zero_filled(raw_csv):
    monthly = aggregate_monthly(load_raw_orders(raw_csv))
    delta = monthly[monthly["store_id"] == "Delta"].set_index("date")["sales"]
    assert delta.loc["2016-03-31"] == 0.0
    # Every store has one row per calendar month, with no gaps.
    for _, group in monthly.groupby("store_id"):
        expected = pd.date_range(group["date"].min(), group["date"].max(), freq="ME")
        assert list(group["date"]) == list(expected)


def test_monthly_totals_match_raw_revenue(raw_csv):
    orders = load_raw_orders(raw_csv)
    monthly = aggregate_monthly(orders, drop_incomplete_last_month=False)
    assert monthly["sales"].sum() == pytest.approx(orders["revenue"].sum())


def test_store_regions(raw_csv):
    regions = store_regions(load_raw_orders(raw_csv)).set_index("store_id")["region"]
    assert regions["Bravo"] == "Europe"
    assert regions.index.is_unique
