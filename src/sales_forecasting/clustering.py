"""Group stores into sales tiers with KMeans."""

from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

PROFILE_COLUMNS = ["avg_sales", "volatility", "max_sales"]


def cluster_labels(n_clusters: int) -> dict[int, str]:
    """Human-readable names for clusters ordered from lowest to highest sales."""
    if n_clusters == 3:
        return {0: "Low", 1: "Medium", 2: "High"}
    return {k: f"Tier {k + 1}" for k in range(n_clusters)}


def store_profile(monthly: pd.DataFrame) -> pd.DataFrame:
    """Per-store summary statistics used as clustering inputs."""
    return (
        monthly.groupby("store_id")["sales"]
        .agg(avg_sales="mean", volatility="std", max_sales="max")
        .fillna(0.0)
        .reset_index()
    )


def fit_store_clusters(
    profile: pd.DataFrame, n_clusters: int = 3, random_state: int = 42
) -> pd.DataFrame:
    """Assign each store a cluster id.

    KMeans ids are arbitrary, so they are renumbered by mean ``avg_sales``:
    cluster 0 always has the lowest-selling stores and the highest id the
    top sellers. That keeps names like "Low"/"High" true across reruns.
    """
    if len(profile) < n_clusters:
        raise ValueError(f"Need at least {n_clusters} stores to form {n_clusters} clusters.")

    scaled = StandardScaler().fit_transform(profile[PROFILE_COLUMNS])
    raw_ids = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit_predict(
        scaled
    )

    result = profile.copy()
    result["cluster"] = raw_ids
    order = result.groupby("cluster")["avg_sales"].mean().sort_values().index
    result["cluster"] = result["cluster"].map({old: new for new, old in enumerate(order)})
    result["cluster_label"] = result["cluster"].map(cluster_labels(n_clusters))
    return result
