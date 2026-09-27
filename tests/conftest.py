"""Shared fixtures: a small synthetic order export shaped like the real one."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

STORES = {
    "Alpha": "Europe",
    "Bravo ": "Europe",  # trailing space, like "Moldova " in the real file
    "Charlie": "Asia",
    "Delta": "Asia",
    "Echo": "Africa",
    "Foxtrot": "Africa",
    "Golf": "Europe",
    "Hotel": "Asia",
    "India": "Africa",
}


def make_orders(seed: int = 0, months: int = 30, last_day: int = 20) -> pd.DataFrame:
    """Raw-format orders from Jan 2015; the final month stops on ``last_day``.

    Store ``Delta`` has no orders in March 2016, to exercise gap filling.
    """
    rng = np.random.default_rng(seed)
    rows = []
    month_starts = pd.date_range("2015-01-01", periods=months, freq="MS")
    for i, (store, region) in enumerate(STORES.items()):
        level = 1_000 * (i + 1)
        for start in month_starts:
            if store == "Delta" and start == pd.Timestamp("2016-03-01"):
                continue
            days = start.days_in_month if start != month_starts[-1] else last_day
            for _ in range(3):
                day = start + pd.Timedelta(days=int(rng.integers(0, days)))
                rows.append(
                    {
                        "Region": region,
                        "Country": store,
                        "Order Date": f"{day.month}/{day.day}/{day.year}",
                        "Total Revenue": round(float(level * rng.uniform(0.5, 1.5)), 2),
                    }
                )
    # Pin the very last order to ``last_day`` so the month end is deterministic.
    rows.append(
        {
            "Region": "Europe",
            "Country": "Alpha",
            "Order Date": f"{month_starts[-1].month}/{last_day}/{month_starts[-1].year}",
            "Total Revenue": 100.0,
        }
    )
    return pd.DataFrame(rows)


@pytest.fixture
def raw_csv(tmp_path):
    path = tmp_path / "orders.csv"
    make_orders().to_csv(path, index=False)
    return path
