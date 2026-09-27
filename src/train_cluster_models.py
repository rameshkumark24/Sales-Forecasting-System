"""Train the cluster-wise forecasting models and write all processed outputs.

Usage (from any directory):
    python src/train_cluster_models.py [--model hgb|rf] [--clusters 3] [--valid-months 6]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from sales_forecasting import config, forecast_next_month, run_training, save_artifacts
from sales_forecasting.evaluation import METHOD_NAMES
from sales_forecasting.modeling import MODEL_KINDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=config.RAW_DATA_PATH, help="raw orders CSV")
    parser.add_argument("--model", choices=sorted(MODEL_KINDS), default=config.DEFAULT_MODEL_KIND)
    parser.add_argument("--clusters", type=int, default=config.N_CLUSTERS)
    parser.add_argument("--valid-months", type=int, default=config.VALIDATION_MONTHS)
    parser.add_argument(
        "--strategy",
        choices=["auto", "cluster", "global"],
        default="auto",
        help="forecast with cluster-wise models, one global model, or whichever scores best",
    )
    parser.add_argument(
        "--keep-partial-month",
        action="store_true",
        help="keep the final month even if the raw data stops before its last day",
    )
    parser.add_argument("--processed-dir", type=Path, default=config.PROCESSED_DIR)
    parser.add_argument("--models-dir", type=Path, default=config.MODELS_DIR)
    return parser.parse_args()


def print_report(metadata: dict) -> None:
    val = metadata["validation"]
    print(
        f"\nHold-out evaluation: {val['start']} to {val['end']} "
        f"({val['months']} months, {val['rows']} store-months)\n"
    )
    print(f"{'Method':<40}{'MAE':>12}{'RMSE':>12}{'WAPE':>8}{'Total err':>11}{'vs naive':>10}")
    overall = metadata["metrics"]["overall"]
    selected = metadata["selected_method"]
    for name, stats in sorted(overall.items(), key=lambda kv: kv[1]["mae"]):
        label = METHOD_NAMES.get(name, name) + (" *" if name == selected else "")
        print(
            f"{label:<40}{stats['mae']:>12,.0f}{stats['rmse']:>12,.0f}{stats['wape']:>8.1%}"
            f"{stats['portfolio_mape']:>11.1%}{stats['improvement_vs_naive_pct']:>9.1f}%"
        )
    print(f"\n* used for the forecast ({metadata['selection']})")
    print("Total err = mean absolute % error of the all-store monthly total")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    artifacts = run_training(
        args.raw,
        model_kind=args.model,
        n_clusters=args.clusters,
        valid_months=args.valid_months,
        drop_incomplete_last_month=not args.keep_partial_month,
        strategy=args.strategy,
    )
    save_artifacts(artifacts, args.processed_dir, args.models_dir)

    forecast = forecast_next_month(artifacts)
    forecast.to_csv(args.processed_dir / config.FORECAST_FILE, index=False, date_format="%Y-%m-%d")

    print_report(artifacts.metadata)
    print(f"\nSaved models to {args.models_dir}")
    print(f"Saved processed data and forecast to {args.processed_dir}")


if __name__ == "__main__":
    main()
