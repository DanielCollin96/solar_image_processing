"""Download SDO solar images from JSOC or the SIDC quicklook archive.

Each run downloads one source, selected with ``--source``. Both sources resume
from the latest date already on disk, so a nightly run only fetches what is
new. Output goes to separate directories:

    JSOC  ->  paths['unprocessed']
    SIDC  ->  paths['unprocessed_realtime']

Usage
-----
::

    uv run scripts/download_solar_images.py --source jsoc               # resume JSOC
    uv run scripts/download_solar_images.py --source sidc               # resume SIDC
    uv run scripts/download_solar_images.py --source sidc --backfill
    uv run scripts/download_solar_images.py --source jsoc --dry-run     # plan only
    uv run scripts/download_solar_images.py --source jsoc --coverage    # report only
"""
import argparse
from datetime import datetime
from pathlib import Path

from solar_image_processing.downloading.solar_image_downloader import (
    SIDCDownloader,
    SolarImageDownloader,
    utc_now,
)
from solar_image_processing.utils.pipeline_config import PipelineConfig

def run_jsoc(downloader, args) -> None:
    """Download JSOC lev1 data at an hourly cadence."""
    downloader.download_images_hourly_cadence(from_start=args.backfill)


def run_sidc(downloader, args) -> None:
    """Download SIDC quicklook data, or crawl the server with --backfill."""
    if args.backfill:
        year, month = (int(part) for part in args.backfill_start.split("-"))
        print(f"Backfilling from {datetime(year, month, 1):%B %Y} ...")
        downloader.backfill(start_year=year, start_month=month)
    else:
        downloader.download_since_latest()


# Available data sources. To add a source, write a run_<name>() function and
# add one entry here; --source picks up the new name automatically.
#   label           name printed in the output
#   downloader      downloader class, constructed with the pipeline config
#   unit            what one planned request is, for the --dry-run summary
#   crawl_backfill  True if --backfill crawls the server instead of a day range
#   run             function that performs the download
SOURCES = {
    "jsoc": {
        "label": "JSOC",
        "downloader": SolarImageDownloader,
        "unit": "export request",
        "crawl_backfill": False,
        "run": run_jsoc,
    },
    "sidc": {
        "label": "SIDC",
        "downloader": SIDCDownloader,
        "unit": "directory listing",
        "crawl_backfill": True,
        "run": run_sidc,
    },
}

def main() -> None:
    parser = argparse.ArgumentParser(
                description="Download SDO solar images from JSOC or SIDC."
    )
    parser.add_argument(
        "--source", choices=list(SOURCES), required=True,
        help="jsoc = JSOC lev1, sidc = SIDC quicklook.",
    )
    parser.add_argument(
        "--backfill", action="store_true",
        help="Ignore what is on disk. SIDC crawls the server month by month; "
             "JSOC re-requests every day from start_date in the configuration "
             "(the full archive).",
    )
    parser.add_argument(
        "--backfill-start", type=str, default="2026-01",
        help="First month to backfill, as YYYY-MM. Only used with --backfill "
             "for SIDC. Default: 2026-01.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show which dates would be requested, then exit. Downloads "
             "nothing and makes no network requests.",
    )
    parser.add_argument(
        "--coverage", action="store_true",
        help="Report the latest downloaded date per channel for the selected "
             "source and exit without downloading anything.",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Config file (default: configs/pipeline_config.yaml).",
    )
    args = parser.parse_args()

    config_path = args.config or (
        Path(__file__).resolve().parent.parent / "configs" / "pipeline_config.yaml"
    )
    config = PipelineConfig(config_path)

    source = SOURCES[args.source]
    label = source["label"]
    downloader = source["downloader"](config)

    if args.coverage:
        print(f"\n{label}  ({downloader.path_downloaded})")
        print(f"  {'channel':<10} {'latest downloaded':<20} resume from")
        for channel in downloader.channels:
            latest = downloader.latest_downloaded(channel)
            resume = downloader.resume_date(
                channel, downloader.RESUME_OFFSET_DAYS
            )
            latest_str = (
                latest.strftime("%Y-%m-%d %H:%M") if latest else "nothing on disk"
            )
            print(f"  {channel:<10} {latest_str:<20} {resume:%Y-%m-%d}")
        print()
        return
    if args.dry_run:
        print("=== DRY RUN - nothing will be downloaded ===")
        print(f"Configuration: {config_path}")
        print(f"Now (UTC):     {utc_now():%Y-%m-%d %H:%M}\n")

        print(f"{label}  ({downloader.path_downloaded})")
        total = 0
        # A crawling backfill (SIDC) starts from --backfill-start and does
        # not walk a day range, so a day plan would be misleading here.
        crawl_backfill = args.backfill and source["crawl_backfill"]

        for channel in downloader.channels:
            latest = downloader.latest_downloaded(channel)
            latest_str = (
                latest.strftime("%Y-%m-%d %H:%M") if latest else "nothing on disk"
            )

            if crawl_backfill:
                print(f"  {channel:<10} latest {latest_str:<20} "
                      f"server crawl from {args.backfill_start}")
                continue

            days = downloader.planned_days(channel, from_start=args.backfill)
            total += len(days)
            if not days:
                print(f"  {channel:<10} latest {latest_str:<20} up to date")
                continue
            print(f"  {channel:<10} latest {latest_str:<20} "
                  f"{days[0]:%Y-%m-%d} -> {days[-1]:%Y-%m-%d}  "
                  f"({len(days)} day(s))")
            if len(days) <= 10:
                print(f"             {', '.join(f'{d:%Y-%m-%d}' for d in days)}")
        if crawl_backfill:
            print("  -> month-by-month server crawl; volume unknown "
                  "until it runs\n")
        else:
            print(f"  -> {total} {source['unit']}(s) in total\n")

        print("Re-run without --dry-run to download.")
        return

    print(f"Source:        {args.source}")
    print(f"Configuration: {config_path}")
    print(f"Channels:      {config.channels}")
    print(f"Now (UTC):     {utc_now():%Y-%m-%d %H:%M}")
    print(f"\n=== {label} ===")

    source["run"](downloader, args)

    print("\nDone.")

if __name__ == "__main__":
    main()
