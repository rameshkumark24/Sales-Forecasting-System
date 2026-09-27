import numpy as np
import pandas as pd
import pytest

from sales_forecasting.clustering import fit_store_clusters, store_profile


def test_cluster_ids_ordered_by_sales():
    rng = np.random.default_rng(1)
    levels = [100] * 5 + [1_000] * 5 + [10_000] * 5
    monthly = pd.DataFrame(
        [
            {"store_id": f"s{i}", "sales": level * rng.uniform(0.9, 1.1)}
            for i, level in enumerate(levels)
            for _ in range(12)
        ]
    )
    clusters = fit_store_clusters(store_profile(monthly), n_clusters=3, random_state=0)
    by_store = clusters.set_index("store_id")
    assert set(by_store.loc[["s0", "s1", "s2", "s3", "s4"], "cluster"]) == {0}
    assert set(by_store.loc[["s10", "s11", "s12", "s13", "s14"], "cluster"]) == {2}
    assert by_store.loc["s0", "cluster_label"] == "Low"
    assert by_store.loc["s14", "cluster_label"] == "High"


def test_too_few_stores():
    profile = pd.DataFrame(
        {"store_id": ["a"], "avg_sales": [1.0], "volatility": [0.0], "max_sales": [1.0]}
    )
    with pytest.raises(ValueError):
        fit_store_clusters(profile, n_clusters=3)
