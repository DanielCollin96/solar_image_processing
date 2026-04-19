"""
Preprocessing runner with two modes:
    catch-up    : Process the full date range from pipeline_config.yaml.
                  Run this once manually to preprocess all existing downloads.
    incremental : Process only the last N days (default 7).
                  Schedule this as a nightly cron job after the downloader.

Usage
-----
    # Catch-up (full range from config)
    uv run python scripts/run_preprocessing.py --mode catch-up

    # Incremental (last 7 days, the default)
    uv run python scripts/run_preprocessing.py --mode incremental

    # Incremental with custom lookback window
    uv run python scripts/run_preprocessing.py --mode incremental --days 14
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path

from solar_image_processing.preprocessing.solar_image_preprocessor import SolarImagePreprocessor
from solar_image_processing.utils.pipeline_config import PipelineConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Run solar image preprocessing in catch-up or incremental mode.'
    )
    parser.add_argument(
        '--mode',
        choices=['catch-up', 'incremental'],
        default='incremental',
        help='catch-up = full date range from config; '
             'incremental = last N days only (default: incremental)',
    )
    parser.add_argument(
        '--days',
        type=int,
        default=7,
        help='Lookback window in days for incremental mode (default: 7)',
    )
    args = parser.parse_args()

    # ---------- Load config ----------
    config_path = Path(__file__).resolve().parent.parent / 'configs' / 'pipeline_config.yaml'
    config = PipelineConfig(config_path)

    # ---------- Override dates for incremental mode ----------
    if args.mode == 'incremental':
        now = datetime.utcnow()
        config.start_date = now - timedelta(days=args.days)
        config.end_date = now
        print(f'[incremental] Processing window: {config.start_date:%Y-%m-%d} → {config.end_date:%Y-%m-%d}')
    else:
        print(f'[catch-up] Processing full range: {config.start_date:%Y-%m-%d} → {config.end_date:%Y-%m-%d}')

    # ---------- Run ----------
    preprocessor = SolarImagePreprocessor(config)
    preprocessor.run()
    print('\nDone.')


if __name__ == '__main__':
    main()
