"""Model factory and persistence."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

MODEL_KINDS = {
    "hgb": "HistGradientBoosting",
    "rf": "RandomForest",
}


def make_model(kind: str = "hgb", random_state: int = 42):
    """Create an unfitted regressor.

    Both options minimise squared error, so they forecast the expected (mean)
    sales. An absolute-error loss scores ~2% better on store-level MAE here,
    but it forecasts the median, and on right-skewed sales that makes the
    all-store monthly total ~17% too low, which is the number planners use.

    ``hgb`` (default) is small and fast to retrain; ``rf`` is the original
    RandomForest approach, kept for comparison.
    """
    if kind == "hgb":
        return HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.02,
            max_iter=150,
            max_leaf_nodes=15,
            min_samples_leaf=40,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=random_state,
        )
    if kind == "rf":
        return RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=20,
            max_features=0.5,
            n_jobs=-1,
            random_state=random_state,
        )
    raise ValueError(f"Unknown model kind {kind!r}; choose from {sorted(MODEL_KINDS)}.")


def save_models(models: dict, metadata: dict, models_dir: Path, bundle_file: str, meta_file: str):
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, models_dir / bundle_file, compress=3)
    with open(models_dir / meta_file, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
        fh.write("\n")


def load_metadata(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python src/train_cluster_models.py`.")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_models(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python src/train_cluster_models.py`.")
    return joblib.load(path)


def runtime_versions() -> dict[str, str]:
    """Library versions that decide whether a pickled model bundle can be loaded."""
    return {"sklearn_version": sklearn.__version__, "numpy_version": np.__version__}


def version_mismatch(metadata: dict) -> str | None:
    """Explain why saved models may be unsafe to load in this environment, if so.

    scikit-learn must match exactly; numpy must share the major version, as
    numpy 1.x cannot unpickle random-state objects written by numpy 2.x.
    """
    saved_sklearn = metadata.get("sklearn_version")
    if saved_sklearn != sklearn.__version__:
        return (
            f"Models were trained with scikit-learn {saved_sklearn} but {sklearn.__version__} "
            "is installed; pickled models are only reliable on the version that created them."
        )
    saved_numpy = metadata.get("numpy_version")
    if saved_numpy and saved_numpy.split(".")[0] != np.__version__.split(".")[0]:
        return (
            f"Models were saved with numpy {saved_numpy} but {np.__version__} is installed; "
            "pickles do not load across numpy major versions."
        )
    return None
