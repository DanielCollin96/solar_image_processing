"""
SIDC Real-Time Solar Image Downloader

Downloads AIA quicklook FITS images from the SIDC/ROB archive at:
    https://sdo.oma.be/data/aia_quicklook/

Data characteristics:
    - Spatial resolution : 1024 x 1024 pixels
    - Cadence            : 3 minutes
    - Latency            : ~15 minutes (near real-time)
    - Availability       : last 2 weeks
    - Wavelengths        : 94, 131, 171, 193, 211, 304, 335, 1600, 1700, 4500 Å

Files are saved into the same directory structure as the JSOC downloader:
    <unprocessed_realtime_path>/AIA/<wavelength>/<YYYY>/<MM>/<DD>/

Usage
-----
    from solar_image_processing.downloading.sidc_downloader import SIDCDownloader
    from solar_image_processing.utils.pipeline_config import PipelineConfig

    config = PipelineConfig('configs/pipeline_config.yaml')
    downloader = SIDCDownloader(config)

    # Download all images in a time window (e.g. last 12 minutes)
    from datetime import datetime, timedelta
    downloader.download_range(
        start=datetime.utcnow() - timedelta(minutes=12),
        end=datetime.utcnow(),
    )
"""

import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path

import astropy.io.fits as fits
import pandas as pd
import requests
from bs4 import BeautifulSoup

from solar_image_processing.utils.pipeline_config import PipelineConfig

# Base URL of the SIDC AIA quicklook archive
_BASE_URL = "https://sdo.oma.be/data/aia_quicklook"

# Only these channels are downloaded for the realtime pipeline
_REALTIME_CHANNELS = {"aia_171", "aia_193", "aia_211"}

# Mapping from pipeline channel name to the zero-padded folder name used by SIDC
_CHANNEL_TO_SIDC_FOLDER = {
    "aia_094": "0094",
    "aia_131": "0131",
    "aia_171": "0171",
    "aia_193": "0193",
    "aia_211": "0211",
    "aia_304": "0304",
    "aia_335": "0335",
    "aia_1600": "1600",
    "aia_1700": "1700",
    "aia_4500": "4500",
}


class SIDCDownloader:
    """
    Download near-real-time AIA quicklook images from the SIDC/ROB archive.

    Parameters
    ----------
    config : PipelineConfig
        Full pipeline configuration object. Only aia_171, aia_193, aia_211
        channels are downloaded for the realtime pipeline.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.channels = [ch for ch in config.channels if ch in _REALTIME_CHANNELS]
        self.path_downloaded = config.paths["unprocessed_realtime"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download_range(self, start: datetime, end: datetime) -> None:
        """
        Download all available images between *start* and *end* (UTC) for
        every configured AIA channel.

        Parameters
        ----------
        start : datetime
            Start of the time window (inclusive, UTC).
        end : datetime
            End of the time window (inclusive, UTC).
        """
        # Warn if start is older than 2 weeks
        two_weeks_ago = datetime.utcnow() - timedelta(weeks=2)
        if start < two_weeks_ago:
            print(f"Warning: SIDC only keeps the last 2 weeks of data.")
            print(f"Your start date {start} is older than {two_weeks_ago.strftime('%Y-%m-%d')}.")
            print(f"Adjusting start date to {two_weeks_ago.strftime('%Y-%m-%d')} ...")
            start = two_weeks_ago

        for channel in self.channels:
            print(f"\n[{channel}] Downloading images from {start} to {end} ...")
            urls = self._list_files_in_range(channel, start, end)

            if not urls:
                print(f"[{channel}] No files found in range — skipping.")
                continue

            for url, filename in urls:
                obs_time = self._parse_time_from_filename(filename)
                save_dir = self._get_save_dir(channel, obs_time)
                filepath = self._download_file(url, save_dir, filename)

                fits_path = save_dir / filename
                pickle_path = save_dir / f"meta_data_{obs_time.strftime('%Y%m%d%H%M')}.pickle"

                if not pickle_path.exists():
                    self._save_metadata(fits_path, save_dir)

    def backfill(self, start_year: int = 2026, start_month: int = 1) -> None:
        """
        Crawl the SIDC server from start_year/start_month to now and download
        everything available that isn't already on disk.

        Parameters
        ----------
        start_year : int
            Year to start backfill from.
        start_month : int
            Month to start backfill from.
        """
        now = datetime.utcnow()

        for channel in self.channels:
            folder = self._sidc_folder(channel)
            print(f"\n[{channel}] Starting backfill ...")

            current = datetime(start_year, start_month, 1)

            while current <= now:
                year_str = current.strftime("%Y")
                month_str = current.strftime("%m")

                # Check which days exist for this month
                month_url = f"{_BASE_URL}/{folder}/{year_str}/{month_str}/"
                try:
                    response = requests.get(month_url, timeout=30)
                    response.raise_for_status()
                except requests.RequestException:
                    print(f"  Skipping {year_str}/{month_str} — not available")
                    if current.month == 12:
                        current = datetime(current.year + 1, 1, 1)
                    else:
                        current = datetime(current.year, current.month + 1, 1)
                    continue

                # Parse available day folders from the listing
                soup = BeautifulSoup(response.text, "html.parser")
                day_folders = [
                    a["href"].strip("/")
                    for a in soup.find_all("a", href=True)
                    if a["href"].strip("/").isdigit() and len(a["href"].strip("/")) == 2
                ]

                print(f"  {year_str}/{month_str} — found days: {sorted(day_folders)}")

                for day_str in sorted(day_folders):
                    day_url = f"{month_url}{day_str}/"
                    filenames = self._list_fits_in_url(day_url)

                    for filename in sorted(filenames):
                        try:
                            obs_time = self._parse_time_from_filename(filename)
                        except ValueError:
                            continue

                        save_dir = self._get_save_dir(channel, obs_time)
                        fits_path = save_dir / filename
                        pickle_path = save_dir / f"meta_data_{obs_time.strftime('%Y%m%d%H%M')}.pickle"

                        self._download_file(day_url + filename, save_dir, filename)

                        if fits_path.exists() and not pickle_path.exists():
                            self._save_metadata(fits_path, save_dir)

                # Move to next month
                if current.month == 12:
                    current = datetime(current.year + 1, 1, 1)
                else:
                    current = datetime(current.year, current.month + 1, 1)

    # ------------------------------------------------------------------
    # Directory helpers
    # ------------------------------------------------------------------

    def _get_save_dir(self, channel: str, obs_time: datetime) -> Path:
        """
        Return (and create) the save directory for a given channel and date.

        Layout:  AIA/<wavelength>/<YYYY>/<MM>/<DD>/
        """
        wavelength = channel.split("_")[-1]  # e.g. '171'
        save_dir = (
            self.path_downloaded
            / "AIA"
            / wavelength
            / obs_time.strftime("%Y")
            / obs_time.strftime("%m")
            / obs_time.strftime("%d")
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir

    # ------------------------------------------------------------------
    # URL / listing helpers
    # ------------------------------------------------------------------

    def _sidc_folder(self, channel: str) -> str:
        """Return the SIDC wavelength folder name for a pipeline channel."""
        folder = _CHANNEL_TO_SIDC_FOLDER.get(channel)
        if folder is None:
            raise ValueError(
                f"Channel '{channel}' is not supported by the SIDC downloader. "
                f"Supported channels: {list(_CHANNEL_TO_SIDC_FOLDER)}"
            )
        return folder

    def _day_url(self, channel: str, date: datetime) -> str:
        """Build the URL for a specific day directory on the SIDC server."""
        folder = self._sidc_folder(channel)
        return (
            f"{_BASE_URL}/{folder}"
            f"/{date.strftime('%Y')}"
            f"/{date.strftime('%m')}"
            f"/{date.strftime('%d')}/"
        )

    def _list_fits_in_url(self, url: str) -> list[str]:
        """
        Parse an Apache-style directory listing and return all .fits filenames.

        Returns an empty list when the page is unreachable or contains no
        FITS files.
        """
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"  Warning: could not reach {url} — {e}")
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        return [
            a["href"]
            for a in soup.find_all("a", href=True)
            if a["href"].endswith(".fits")
        ]

    def _list_files_in_range(
        self,
        channel: str,
        start: datetime,
        end: datetime,
    ) -> list[tuple[str, str]]:
        """
        Return a list of (url, filename) pairs for all FITS files whose
        observation time falls within [start, end].
        """
        results = []

        current_day = start.replace(hour=0, minute=0, second=0, microsecond=0)
        while current_day <= end:
            day_url = self._day_url(channel, current_day)
            filenames = self._list_fits_in_url(day_url)

            for filename in sorted(filenames):
                try:
                    obs_time = self._parse_time_from_filename(filename)
                except ValueError:
                    continue

                if start <= obs_time <= end:
                    results.append((day_url + filename, filename))

            current_day += timedelta(days=1)

        return results

    # ------------------------------------------------------------------
    # Filename / time helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_time_from_filename(filename: str) -> datetime:
        """
        Extract the observation datetime from a SIDC quicklook filename.

        Expected format::

            aia_quicklook.0171.20260201_000900.fits
                                 ^^^^^^^^ ^^^^^^
                                 date     time (HHMMSS)

        Returns
        -------
        datetime
            Parsed UTC observation time.

        Raises
        ------
        ValueError
            If the filename does not match the expected pattern.
        """
        try:
            parts = filename.split(".")
            date_time_str = parts[2]
            return datetime.strptime(date_time_str, "%Y%m%d_%H%M%S")
        except (IndexError, ValueError) as exc:
            raise ValueError(
                f"Cannot parse observation time from filename '{filename}'"
            ) from exc

    # ------------------------------------------------------------------
    # Download helper
    # ------------------------------------------------------------------

    @staticmethod
    def _download_file(url: str, save_dir: Path, filename: str) -> Path | None:
        """
        Download a single FITS file from *url* and save it to *save_dir*.

        Skips the download if the file already exists.

        Returns
        -------
        Path | None
            Path to the downloaded file, or None if skipped.
        """
        dest = save_dir / filename
        if dest.exists():
            print(f"  Already exists — skipping: {filename}")
            return None

        print(f"  Downloading: {filename} ...", end=" ", flush=True)
        start = time.time()
        try:
            response = requests.get(url, timeout=120, stream=True)
            response.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            elapsed = time.time() - start
            size_kb = dest.stat().st_size / 1024
            print(f"done ({size_kb:.0f} KB, {elapsed:.1f}s)")
            return dest
        except requests.RequestException as e:
            print(f"FAILED — {e}")
            if dest.exists():
                dest.unlink()
            return None

    # ------------------------------------------------------------------
    # Metadata helper
    # ------------------------------------------------------------------

    def _save_metadata(self, filepath: Path, save_dir: Path) -> None:
        """
        Extract FITS header metadata and save as a pickle file.

        Reads the header from the downloaded FITS file and saves it as a
        pandas DataFrame pickle, following the same convention as the
        JSOC downloader.

        Also called when the FITS file already exists but the pickle is
        missing — ensuring no metadata gaps.
        """
        try:
            with fits.open(filepath) as hdul:
                header = hdul[1].header
                metadata = {key: [val] for key, val in header.items()}
                df = pd.DataFrame(metadata)

            obs_time = self._parse_time_from_filename(filepath.name)
            pickle_path = save_dir / f"meta_data_{obs_time.strftime('%Y%m%d%H%M')}.pickle"
            with open(pickle_path, "wb") as f:
                pickle.dump(df, f)
            print(f"  Metadata saved: {pickle_path.name}")
        except Exception as e:
            print(f"  Warning: could not save metadata for {filepath.name} — {e}")
