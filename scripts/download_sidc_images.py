"""
Script to download near-real-time AIA quicklook images from the SIDC archive.

Edit ``configs/pipeline_config.yaml`` to configure which channels to download.

Usage
-----
Run from the project root:

    uv run python scripts/download_sidc_images.py             # last 12 minutes (default)
    uv run python scripts/download_sidc_images.py --hours 6   # last 6 hours
    uv run python scripts/download_sidc_images.py --hours 48  # last 2 days
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path

from solar_image_processing.downloading.sidc_downloader import SIDCDownloader
from solar_image_processing.utils.pipeline_config import PipelineConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download near-real-time AIA images from SIDC."
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=None,
        help=(
            "Download all images from the last N hours. "
            "Omit this flag to download the last 12 minutes (default for cron)."
        ),
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Crawl SIDC server and download all historically available files."
    )
    args = parser.parse_args()

    config_path = Path.cwd() / "configs" / "pipeline_config.yaml"
    config = PipelineConfig(config_path)
    downloader = SIDCDownloader(config)

    if args.backfill:
        print("Starting backfill from January 2026 ...")
        downloader.backfill(start_year=2026, start_month=1)
    else:
        end = datetime.utcnow()
        if args.hours is None:
            start = end - timedelta(minutes=12)
            print(f"Downloading images from last 12 minutes ({start:%H:%M} - {end:%H:%M} UTC) ...")
        else:
            start = end - timedelta(hours=args.hours)
            print(f"Downloading images from {start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M} UTC ...")
        downloader.download_range(start=start, end=end)

    print("\nDone.")

    if args.hours is None:
        # Default: last 12 minutes — used by cron job
        # This window is larger than the 3-minute cadence to ensure
        # no images are missed if the cron job runs slightly late
        start = end - timedelta(minutes=20)
        print(f"Downloading images from last 20 minutes ({start:%H:%M} - {end:%H:%M} UTC) ...")
    else:
        start = end - timedelta(hours=args.hours)
        print(f"Downloading images from {start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M} UTC ...")

    downloader.download_range(start=start, end=end)
    print("\nDone.")


if __name__ == "__main__":
    main()
