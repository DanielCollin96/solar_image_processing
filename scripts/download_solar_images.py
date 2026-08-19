"""
Download SDO solar images from JSOC and/or the SIDC quicklook archive.

Both sources resume from the latest date already on disk, so a nightly run only
fetches what is new. Output goes to separate directories:

    JSOC  ->  paths['unprocessed']
    SIDC  ->  paths['unprocessed_realtime']

Usage
-----
Run from the project root::

    uv run python scripts/download_solar_images.py                  # both sources, resume
    uv run python scripts/download_solar_images.py --jsoc           # JSOC only
    uv run python scripts/download_solar_images.py --sidc           # SIDC only
    uv run python scripts/download_solar_images.py --backfill --sidc
    uv run python scripts/download_solar_images.py --coverage       # report only

Notes
-----
All settings come from ``configs/pipeline_config.yaml``: channels, start date,
email, rebin factor and output paths.

``--backfill`` ignores what is on disk. For SIDC it crawls the server month by
month; for JSOC it starts from ``start_date`` in the configuration.

JSOC publishes about a week behind real time, so a nightly run requests days
that are not yet available. Those return nothing and are retried the next night.
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download SDO solar images from JSOC and/or SIDC."
    )
    parser.add_argument(
        "--jsoc", action="store_true",
        help="Download JSOC science data only.",
    )
    parser.add_argument(
        "--sidc", action="store_true",
        help="Download SIDC quicklook data only.",
    )
    parser.add_argument(
        "--backfill", action="store_true",
        help="Ignore what is on disk. SIDC crawls the server; JSOC starts from "
             "start_date in the configuration.",
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
        help="Report the latest downloaded date per channel and exit without "
             "downloading anything.",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Path to pipeline_config.yaml. Defaults to "
             "<project root>/configs/pipeline_config.yaml.",
    )
    args = parser.parse_args()

    config_path = args.config or Path.cwd() / "configs" / "pipeline_config.yaml"
    config = PipelineConfig(config_path)

    # Neither flag given means both sources.
    do_jsoc = args.jsoc or not (args.jsoc or args.sidc)
    do_sidc = args.sidc or not (args.jsoc or args.sidc)

    # --backfill is destructive for JSOC: it ignores what is on disk and
    # restarts from start_date, re-requesting the entire archive. Require an
    # explicit source so it can never be triggered by accident.
    if args.backfill and not (args.jsoc or args.sidc):
        parser.error(
            "--backfill needs an explicit source: --sidc or --jsoc.\n"
            "  --backfill --sidc  crawls the SIDC server month by month\n"
            "  --backfill --jsoc  re-requests EVERY day from start_date "
            "(the full archive)"
        )

    if args.coverage:
        for label, downloader in (
            ("JSOC", SolarImageDownloader(config)),
            ("SIDC", SIDCDownloader(config)),
        ):
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

        sources = []
        if do_jsoc:
            sources.append(("JSOC", SolarImageDownloader(config), "export request"))
        if do_sidc:
            sources.append(("SIDC", SIDCDownloader(config), "directory listing"))

        for label, downloader, unit in sources:
            print(f"{label}  ({downloader.path_downloaded})")
            total = 0
            # SIDC backfill crawls the server from --backfill-start; it does
            # not walk a day range, so a day plan would be misleading here.
            sidc_backfill = args.backfill and label == "SIDC"

            for channel in downloader.channels:
                latest = downloader.latest_downloaded(channel)
                latest_str = (
                    latest.strftime("%Y-%m-%d %H:%M") if latest else "nothing on disk"
                )

                if sidc_backfill:
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
            if sidc_backfill:
                print("  -> month-by-month server crawl; volume unknown "
                      "until it runs\n")
            else:
                print(f"  -> {total} {unit}(s) in total\n")

        print("Re-run without --dry-run to download.")
        return

    print(f"Configuration: {config_path}")
    print(f"Channels:      {config.channels}")
    print(f"Now (UTC):     {utc_now():%Y-%m-%d %H:%M}")

    if do_jsoc:
        print("\n=== JSOC ===")
        SolarImageDownloader(config).download_images_hourly_cadence(
            from_start=args.backfill
        )

    if do_sidc:
        print("\n=== SIDC ===")
        downloader = SIDCDownloader(config)
        if args.backfill:
            year, month = (int(part) for part in args.backfill_start.split("-"))
            print(f"Backfilling from {datetime(year, month, 1):%B %Y} ...")
            downloader.backfill(start_year=year, start_month=month)
        else:
            downloader.download_since_latest()

    print("\nDone.")


if __name__ == "__main__":
    main()
