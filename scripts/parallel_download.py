#!/usr/bin/env python3
"""
Parallel Solar Image Downloader — Multi-Email Worker.

Reads pipeline_config.yaml, assigns each worker (email) its designated
channels, and launches one process per worker. Each process auto-resumes
from the latest downloaded date for every channel it owns.

Usage:
    uv run python scripts/parallel_download.py
    uv run python scripts/parallel_download.py --config /path/to/pipeline_config.yaml
    uv run python scripts/parallel_download.py --dry-run
"""

import argparse
import logging
import multiprocessing as mp
import re
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Filename date parsers
# ---------------------------------------------------------------------------

PATTERN_NEW = re.compile(r'\.(\d{4}-\d{2}-\d{2}T\d{6}Z)\.')
PATTERN_OLD = re.compile(r'_(\d{4}_\d{2}_\d{2})t(\d{2})_(\d{2})_(\d{2})', re.IGNORECASE)
PATTERN_QL  = re.compile(r'\.(\d{8}_\d{6})\.')
PATTERN_HMI = re.compile(r'\.(\d{8}_\d{6})_TAI\.')

def parse_date_from_filename(filename: str) -> datetime | None:
    m = PATTERN_NEW.search(filename)
    if m:
        return datetime.strptime(m.group(1), "%Y-%m-%dT%H%M%SZ")
    m = PATTERN_OLD.search(filename)
    if m:
        date_part = m.group(1).replace("_", "")
        hh, mm, ss = m.group(2), m.group(3), min(int(m.group(4)), 59)
        return datetime.strptime(f"{date_part}{hh}{mm}{ss:02d}", "%Y%m%d%H%M%S")
    m = PATTERN_QL.search(filename)
    if m:
        return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
    m = PATTERN_HMI.search(filename)
    if m:
        return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
    return None


def get_latest_date(wavelength_dir: Path) -> datetime | None:
    """Scan all FITS files under a directory and return the latest date."""
    latest = None
    for fits_file in wavelength_dir.rglob("*.fits"):
        dt = parse_date_from_filename(fits_file.name)
        if dt and (latest is None or dt > latest):
            latest = dt
    return latest


def get_data_dir(base_dir: Path, unprocessed_rel: str, channel: str) -> Path:
    """Return the FITS storage directory for a given channel."""
    sdo_root = base_dir / unprocessed_rel
    wl = channel.split("_")[1] if channel.startswith("aia_") else None
    if wl:
        return sdo_root / "AIA" / wl
    return sdo_root / "HMI" / "magnetogram"


def determine_resume_date(
    data_dir: Path,
    config_start: datetime,
    channel: str,
    logger: logging.Logger,
) -> datetime:
    """
    Return the date to resume downloading from.
    - No files yet  -> config start_date
    - Files exist   -> latest date + 1 day
    """
    if not data_dir.exists():
        logger.info(f"[{channel}] No directory - starting from {config_start.date()}")
        return config_start

    latest = get_latest_date(data_dir)
    if latest is None:
        logger.info(f"[{channel}] No FITS files - starting from {config_start.date()}")
        return config_start

    resume = datetime(latest.year, latest.month, latest.day) + timedelta(days=1)
    logger.info(f"[{channel}] Latest: {latest.strftime('%Y-%m-%d %H:%M UTC')} "
                f"-> resuming from {resume.date()}")
    return resume


# ---------------------------------------------------------------------------
# Worker process
# ---------------------------------------------------------------------------

def worker_process(
    worker_index: int,
    email: str,
    channels: list,
    raw_config: dict,
    base_dir: Path,
    config_path: Path,
    log_dir: Path,
    dry_run: bool,
) -> None:
    """
    Runs in a child process. For each assigned channel, determines the
    resume date, then constructs a per-channel PipelineConfig and runs
    the downloader.
    """
    # ---- logging ----
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_email = email.replace("@", "_at_").replace(".", "_")
    log_path = log_dir / f"download_worker_{safe_email}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [worker-%(process)d] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger(f"worker_{worker_index}")

    logger.info("=" * 60)
    logger.info(f"Worker {worker_index} starting  |  email: {email}")
    logger.info(f"Channels: {channels}")
    logger.info(f"Log: {log_path}")
    logger.info("=" * 60)

    # ---- add src to path so imports work in child process ----
    src_path = str(base_dir / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    from solar_image_processing.utils.pipeline_config import PipelineConfig
    from solar_image_processing.downloading.solar_image_downloader import SolarImageDownloader

    config_start = datetime.strptime(
        raw_config.get("start_date", "2010-05-01 00:00:00"), "%Y-%m-%d %H:%M:%S"
    )
    unprocessed_rel = raw_config["paths"]["unprocessed"]

    for channel in channels:
        logger.info(f"\n--- Channel: {channel} ---")

        data_dir = get_data_dir(base_dir, unprocessed_rel, channel)
        resume_date = determine_resume_date(data_dir, config_start, channel, logger)

        if resume_date >= datetime.today():
            logger.info(f"[{channel}] Already up to date - skipping.")
            continue

        if dry_run:
            logger.info(f"[{channel}] DRY RUN - would download "
                        f"from {resume_date.date()} to today.")
            continue

        # Build a temporary yaml with this worker's email, channel,
        # and the correct resume date, then pass it to PipelineConfig
        worker_config = {
            "base_dir":   str(base_dir),
            "paths":      raw_config["paths"],
            "channels":   [channel],
            "start_date": resume_date.strftime("%Y-%m-%d %H:%M:%S"),
            "end_date":   datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
            "download": {
                "email":        email,
                "rebin_factor": raw_config["download"]["rebin_factor"],
                "jsoc_series":  raw_config["download"]["jsoc_series"],
            },
            "preprocessing": raw_config.get("preprocessing", {}),
            "cropping":      raw_config.get("cropping", {}),
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as tmp:
            yaml.dump(worker_config, tmp)
            tmp_path = Path(tmp.name)

        try:
            pipeline_cfg = PipelineConfig(tmp_path)
            downloader = SolarImageDownloader(pipeline_cfg)
            downloader.download_images_hourly_cadence()
            logger.info(f"[{channel}] Finished successfully.")
        except Exception as e:
            logger.error(f"[{channel}] ERROR: {e} - skipping.")
        finally:
            tmp_path.unlink(missing_ok=True)

    logger.info(f"\nWorker {worker_index} ({email}) completed all channels.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel solar image downloads across multiple JSOC emails."
    )
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to pipeline_config.yaml.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview resume dates without downloading.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ---- locate config ----
    script_dir = Path(__file__).resolve().parent
    if args.config:
        config_path = args.config.resolve()
    else:
        config_path = script_dir.parent / "configs" / "pipeline_config.yaml"
        if not config_path.exists():
            config_path = script_dir / "pipeline_config.yaml"

    if not config_path.exists():
        print("ERROR: pipeline_config.yaml not found. Use --config to specify path.")
        sys.exit(1)

    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    base_dir = Path(raw_config["base_dir"]).resolve() if raw_config.get("base_dir") \
               else config_path.parent.parent
    log_dir  = base_dir / "logs"

    workers = raw_config.get("download_workers", [])
    if not workers:
        print("ERROR: No 'download_workers' defined in pipeline_config.yaml.")
        sys.exit(1)

    # ---- sanity check: no channel in more than one worker ----
    seen = {}
    for w in workers:
        for ch in w["channels"]:
            if ch in seen:
                print(f"ERROR: '{ch}' assigned to both '{seen[ch]}' and '{w['email']}'.")
                sys.exit(1)
            seen[ch] = w["email"]

    print(f"\nConfig:   {config_path}")
    print(f"Base dir: {base_dir}")
    print(f"Workers:  {len(workers)}")
    for i, w in enumerate(workers):
        print(f"  Worker {i+1}: {w['email']}  ->  {w['channels']}")
    if args.dry_run:
        print("\nDRY RUN MODE - no files will be downloaded.")
    print()

    # ---- spawn one process per worker ----
    processes = []
    for i, w in enumerate(workers):
        p = mp.Process(
            target=worker_process,
            name=f"worker-{i+1}",
            kwargs=dict(
                worker_index=i + 1,
                email=w["email"],
                channels=w["channels"],
                raw_config=raw_config,
                base_dir=base_dir,
                config_path=config_path,
                log_dir=log_dir,
                dry_run=args.dry_run,
            ),
        )
        p.start()
        print(f"Launched worker {i+1} (PID {p.pid})  ->  {w['email']}")
        processes.append(p)

    print("\nWaiting for all workers to finish...")
    for p in processes:
        p.join()
        status = "OK" if p.exitcode == 0 else f"EXIT CODE {p.exitcode}"
        print(f"{p.name} finished - {status}")

    print("\nAll workers done.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
