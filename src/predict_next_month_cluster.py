"""Forecast next-month sales for every store with the saved models.

Usage (from any directory, after training):
    python src/predict_next_month_cluster.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from sales_forecasting import (
    IncompatibleModelsError,
    config,
    forecast_next_month,
    load_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processed-dir", type=Path, default=config.PROCESSED_DIR)
    parser.add_argument("--models-dir", type=Path, default=config.MODELS_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"output CSV (default: <processed-dir>/{config.FORECAST_FILE})",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    try:
        artifacts = load_artifacts(args.processed_dir, args.models_dir)
    except (FileNotFoundError, IncompatibleModelsError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    forecast = forecast_next_month(artifacts)
    output = args.output or args.processed_dir / config.FORECAST_FILE
    output.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(output, index=False, date_format="%Y-%m-%d")

    month = forecast["forecast_month"].max().strftime("%B %Y")
    total = forecast["forecast_sales"].sum()
    last_total = forecast["last_month_sales"].sum()
    print(f"\nForecast for {month}: {len(forecast)} stores")
    print(f"Total forecast: ${total:,.0f} (last month: ${last_total:,.0f})")
    print("\nSample:")
    print(forecast.head().to_string(index=False))
    print(f"\nSaved forecast to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
