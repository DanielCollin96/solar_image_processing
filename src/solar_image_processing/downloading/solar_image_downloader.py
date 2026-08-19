"""
Solar image downloaders.

Two sources, one module:

``SolarImageDownloader``
    Science-quality AIA EUV images from JSOC, via the ``drms`` export client.
    Hourly cadence, one batch request per day with a per-hour fallback.
    Written to ``paths['unprocessed']``.

``SIDCDownloader``
    Near-real-time AIA quicklook images from the SIDC/ROB archive, via HTTP
    directory listings. Written to ``paths['unprocessed_realtime']``.

Both write the same day-wise layout::

    <root>/AIA/<wavelength>/<YYYY>/<MM>/<DD>/

and both resume from the latest date already on disk rather than rescanning the
whole configured range. Only AIA is downloaded; HMI is no longer fetched from
either source.

All times are UTC. Filenames carry UTC observation times, so local time is never
used for bounds or comparisons.
"""

import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import astropy.io.fits as fits
import pandas as pd
import requests
from bs4 import BeautifulSoup

from solar_image_processing.downloading import jsoc_download as jd
from solar_image_processing.utils.pipeline_config import PipelineConfig

# Base URL of the SIDC AIA quicklook archive
_BASE_URL = "https://sdo.oma.be/data/aia_quicklook"

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

# Only AIA is downloaded. HMI files already on disk remain readable.
_INSTRUMENT_DIR = "AIA"


def utc_now() -> datetime:
    """
    Return the current UTC time as a naive datetime.

    Observation times parsed from filenames are naive UTC, so bounds and
    comparisons must use UTC too. Using local time here would put the end bound
    ahead of real time by the server's UTC offset.
    """
    return datetime.utcnow()


class BaseDownloader:
    """
    Shared behaviour for both sources: directory layout, resume, and metadata.

    Subclasses set :attr:`path_downloaded` and implement their own download
    logic. This class holds only what both genuinely share.

    Parameters
    ----------
    config : PipelineConfig
        Full pipeline configuration object.
    path_key : str
        Key into ``config.paths`` naming this source's download root, e.g.
        ``'unprocessed'`` or ``'unprocessed_realtime'``.
    """

    def __init__(self, config: PipelineConfig, path_key: str) -> None:
        self.config = config
        self.channels = [ch for ch in config.channels if ch.startswith("aia")]
        self.start_date = config.start_date
        self.path_downloaded: Path = config.paths[path_key]

    #: Days to add to the latest downloaded day when resuming. Subclasses
    #: override: 1 never re-requests a day, 0 restarts on the latest day.
    RESUME_OFFSET_DAYS = 1

    def planned_days(self, channel: str, from_start: bool = False) -> List[datetime]:
        """
        Return the list of days a run would request for *channel*.

        Computed from local disk only -- no network access -- so it is safe and
        instant to call for a dry run.

        Parameters
        ----------
        from_start : bool, optional
            If ``True``, plan from ``config.start_date`` instead of resuming.

        Returns
        -------
        list of datetime
            Midnight of each day that would be requested, oldest first. Empty if
            the channel is already up to date.
        """
        end = utc_now()
        start = (
            self.start_date
            if from_start
            else self.resume_date(channel, self.RESUME_OFFSET_DAYS)
        )

        days = []
        day = datetime(start.year, start.month, start.day)
        while day < end:
            days.append(day)
            day += timedelta(days=1)
        return days


    # ------------------------------------------------------------------
    # Directory layout
    # ------------------------------------------------------------------

    def _channel_root(self, channel: str) -> Path:
        """
        Return the per-channel root directory, ``<root>/AIA/<wavelength>``.

        Raises
        ------
        ValueError
            If *channel* is not an AIA channel. HMI is no longer downloaded, and
            an unrecognised channel should fail loudly rather than silently
            writing somewhere unexpected.
        """
        if not channel.startswith("aia"):
            raise ValueError(
                f"Only AIA channels can be downloaded, got '{channel}'. "
                "HMI is no longer fetched from either source."
            )
        return self.path_downloaded / _INSTRUMENT_DIR / channel.split("_")[-1]

    def _get_day_path(self, channel: str, date: datetime, create: bool = True) -> Path:
        """Return (and optionally create) ``<channel root>/<YYYY>/<MM>/<DD>``."""
        day_path = (
            self._channel_root(channel)
            / date.strftime("%Y")
            / date.strftime("%m")
            / date.strftime("%d")
        )
        if create:
            day_path.mkdir(parents=True, exist_ok=True)
        return day_path

    # ------------------------------------------------------------------
    # Resume: what is already on disk?
    #
    # The archive holds millions of files, so globbing every FITS file is not
    # viable. The day-wise directory names are themselves an index: descending
    # newest-first and stopping at the first day that actually contains a
    # readable file costs three directory listings plus one small one,
    # regardless of how large the archive grows.
    # ------------------------------------------------------------------

    @staticmethod
    def _numeric_subdirs(path: Path) -> List[Path]:
        """Return numerically-named subdirectories of *path*, sorted ascending."""
        if not path.is_dir():
            return []
        return sorted(
            (p for p in path.iterdir() if p.is_dir() and p.name.isdigit()),
            key=lambda p: p.name,
        )

    def _iter_day_dirs_newest_first(self, root: Path):
        """Yield ``<YYYY>/<MM>/<DD>`` directories under *root*, newest first."""
        for year in reversed(self._numeric_subdirs(root)):
            for month in reversed(self._numeric_subdirs(year)):
                for day in reversed(self._numeric_subdirs(month)):
                    yield day

    @staticmethod
    def _latest_in_day(day_dir: Path) -> Optional[datetime]:
        """
        Return the newest observation time among FITS files in one day directory.

        Unparseable filenames are skipped rather than raising, so a stray file
        (a README, a leftover tar, a partial download) cannot break resume for a
        whole channel.
        """
        # Imported lazily: helper_functions pulls in sunpy and aiapy, which the
        # download path does not otherwise need.
        from solar_image_processing.utils.helper_functions import read_file_name

        latest = None
        for file in day_dir.iterdir():
            if file.suffix != ".fits":
                continue
            try:
                file_date, _, _, _ = read_file_name(file.name)
            except Exception:
                continue
            if latest is None or file_date > latest:
                latest = file_date
        return latest

    def latest_downloaded(self, channel: str) -> Optional[datetime]:
        """
        Return the newest observation time on disk for *channel*.

        Descends newest-first and stops at the first day directory holding a
        parseable FITS file, so empty day folders left behind by a failed
        download are skipped automatically.

        Returns
        -------
        datetime or None
            ``None`` if nothing has been downloaded for this channel yet.
        """
        for day_dir in self._iter_day_dirs_newest_first(self._channel_root(channel)):
            latest = self._latest_in_day(day_dir)
            if latest is not None:
                return latest
        return None

    def resume_date(self, channel: str, offset_days: int) -> datetime:
        """
        Return the date to start downloading from for *channel*.

        Parameters
        ----------
        offset_days : int
            Days to add to the latest downloaded day. ``1`` starts on the day
            after (never re-requests a day already on disk); ``0`` restarts at
            midnight of the latest day, which lets a source that skips existing
            files complete a partly-downloaded day.

        Returns
        -------
        datetime
            Midnight of the resume day, or ``config.start_date`` if this channel
            has nothing on disk yet.
        """
        latest = self.latest_downloaded(channel)
        if latest is None:
            return self._fallback_start()
        midnight = datetime(latest.year, latest.month, latest.day)
        return midnight + timedelta(days=offset_days)

    def _fallback_start(self) -> datetime:
        """
        Return the start date used when a channel has nothing on disk.

        Defaults to ``config.start_date``. Overridden by sources whose archive
        does not reach back that far.
        """
        return self.start_date

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @staticmethod
    def _write_pickle(obj, filepath: Path) -> None:
        """Pickle *obj* to *filepath*."""
        with open(filepath, "wb") as fh:
            pickle.dump(obj, fh)


class SolarImageDownloader(BaseDownloader):
    """
    Download science-quality AIA images from JSOC at hourly cadence.

    For each day a single batch request is attempted first; if it fails, the
    day is retried one hour at a time. Downloads resume from the day after the
    latest date already on disk.

    Parameters
    ----------
    config : PipelineConfig
        Full pipeline configuration object.

    Notes
    -----
    JSOC exports a tar archive and extracts it without checking for existing
    files, so re-requesting a day means re-downloading it. Resume therefore
    starts on the day *after* the latest one on disk.

    JSOC publishes roughly a week behind real time. A nightly run will request
    days the archive has not published yet; those return nothing and are simply
    re-attempted the next night, so the pending window closes itself as data
    lands.
    """

    #: Never re-request a day already on disk.
    RESUME_OFFSET_DAYS = 1

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(config, path_key="unprocessed")
        self.download_config: Dict = config.download_config

    def download_images_hourly_cadence(self, from_start: bool = False) -> None:
        """
        Download images at hourly cadence for all configured channels.

        Parameters
        ----------
        from_start : bool, optional
            If ``True``, ignore what is on disk and start from
            ``config.start_date``. Default ``False`` (resume).
        """
        end_date = utc_now()

        for channel in self.channels:
            if from_start:
                current_date = self.start_date
            else:
                current_date = self.resume_date(channel, self.RESUME_OFFSET_DAYS)

            print(f"\n[{channel}] Downloading from {current_date:%Y-%m-%d} "
                  f"to {end_date:%Y-%m-%d} (UTC)")

            if current_date >= end_date:
                print(f"[{channel}] Already up to date - nothing to do.")
                continue

            while current_date < end_date:
                start_time = time.time()
                print(f"Requesting images for {current_date}")

                daily_batch_end_date = min(
                    current_date + timedelta(hours=24), end_date
                )
                day_path = self._get_day_path(channel, current_date)

                # Attempt batch download for the entire day
                success = self._download_daily_batch(
                    current_date, daily_batch_end_date, day_path, channel
                )

                if not success:
                    # Fall back to downloading individual hours
                    self._download_hourly_fallback(current_date, day_path, channel)

                elapsed = time.time() - start_time
                print(f"--- {elapsed:.2f} seconds ---")

                current_date = daily_batch_end_date

    def _download_daily_batch(
        self,
        start_date: datetime,
        end_date: datetime,
        day_path: Path,
        channel: str,
    ) -> bool:
        """
        Attempt to download a full day of images in a single batch request.

        Returns
        -------
        bool
            ``True`` if the download succeeded, ``False`` otherwise.
        """
        client = jd.client(self.download_config["email"])
        series = self.download_config["jsoc_series"][channel[:3]]

        request_string = client.create_request_string(
            series["series"],
            start_date,
            endtime=end_date,
            wavelength=channel[-3:],
            segment=series["segment"],
            period="",
            cadence=timedelta(hours=1),
        )
        print(request_string)
        search_results = client.search(request_string, keys=["t_obs", "**ALL**"])
        print("Request successful. Meta data:")
        print(search_results)
        print("Start downloading.")

        self._save_metadata(search_results, day_path, start_date, "%Y%m%d")

        try:
            client.download(
                request_string,
                str(day_path),
                method="url-tar",
                protocol="fits",
                filter=None,
                rebin=self.download_config["rebin_factor"],
                process={},
            )
            print("Files downloaded successfully.")
            return True
        except Exception as e:
            print(f"File download error for {start_date}: {e}")
            print("Trying hourly downloads as fallback.")
            return False

    def _download_hourly_fallback(
        self,
        start_date: datetime,
        day_path: Path,
        channel: str,
    ) -> None:
        """
        Download images one hour at a time as a fallback strategy.

        Called when the daily batch download fails. Each hour is requested
        independently; failed hours are skipped with a printed warning.
        """
        current_hour = start_date

        for _ in range(24):
            print(f"Requesting single hour: {current_hour}")
            end_hour = current_hour + timedelta(hours=1)

            client = jd.client(self.download_config["email"])
            series = self.download_config["jsoc_series"][channel[:3]]

            request_string = client.create_request_string(
                series["series"],
                current_hour,
                endtime=end_hour,
                wavelength=channel[-3:],
                segment=series["segment"],
                period="",
                cadence=timedelta(hours=1),
            )

            search_results = client.search(request_string, keys=["t_obs", "**ALL**"])
            print("Request for single hour successful. Start downloading.")

            self._save_metadata(search_results, day_path, current_hour, "%Y%m%d%H")

            try:
                client.download(
                    request_string,
                    str(day_path),
                    method="url-tar",
                    protocol="fits",
                    filter=None,
                    rebin=self.download_config["rebin_factor"],
                    process={},
                )
                print("Single file downloaded successfully.")
            except Exception as e:
                print(f"Single file download error for {current_hour}: {e}")
                print("Skipping this hour.")

            current_hour = current_hour + timedelta(hours=1)

    def _save_metadata(
        self,
        search_results: pd.DataFrame,
        day_path: Path,
        date: datetime,
        date_format: str,
    ) -> None:
        """
        Save JSOC search result metadata to a pickle file.

        Parameters
        ----------
        date_format : str
            ``strftime`` format applied to *date* for the filename
            (``'%Y%m%d'`` for daily, ``'%Y%m%d%H'`` for hourly).
        """
        filepath = day_path / f"meta_data_{date.strftime(date_format)}.pickle"
        self._write_pickle(search_results, filepath)


class SIDCDownloader(BaseDownloader):
    """
    Download near-real-time AIA quicklook images from the SIDC/ROB archive.

    Data characteristics:
        - Spatial resolution : 1024 x 1024 pixels
        - Cadence            : 3 minutes
        - Latency            : ~15 minutes
        - Wavelengths        : 94, 131, 171, 193, 211, 304, 335, 1600, 1700, 4500

    Parameters
    ----------
    config : PipelineConfig
        Full pipeline configuration object. Channels are taken from
        ``config.channels`` -- this class no longer maintains its own channel
        list, so the configuration is the single place channels are set.

    Notes
    -----
    Existing files are skipped per file, not per day: every day in the requested
    range is re-listed from the server and only absent files are fetched. A
    partly-downloaded day therefore completes itself, which is why resume starts
    at midnight of the latest day rather than the day after.
    """

    #: Restart on the latest day itself so partial days fill in.
    RESUME_OFFSET_DAYS = 0

    #: How far back to start when a channel has nothing on disk. The SIDC
    #: archive only holds recent data, so falling back to the pipeline
    #: start_date (2010) would issue thousands of listings for days that never
    #: existed. Use --backfill to reach further back deliberately.
    DEFAULT_LOOKBACK_DAYS = 14

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(config, path_key="unprocessed_realtime")

    def _fallback_start(self) -> datetime:
        """Start a fresh channel from the recent past, not from start_date."""
        now = utc_now()
        midnight = datetime(now.year, now.month, now.day)
        return midnight - timedelta(days=self.DEFAULT_LOOKBACK_DAYS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download_since_latest(self) -> None:
        """Download from the latest date on disk up to now, per channel."""
        end = utc_now()

        for channel in self.channels:
            start = self.resume_date(channel, self.RESUME_OFFSET_DAYS)
            print(f"\n[{channel}] Downloading from {start:%Y-%m-%d %H:%M} "
                  f"to {end:%Y-%m-%d %H:%M} (UTC)")
            self._download_channel_range(channel, start, end)

    def download_range(self, start: datetime, end: datetime) -> None:
        """
        Download all available images between *start* and *end* (UTC) for every
        configured channel.
        """
        for channel in self.channels:
            print(f"\n[{channel}] Downloading images from {start} to {end} ...")
            self._download_channel_range(channel, start, end)

    def backfill(self, start_year: int, start_month: int) -> None:
        """
        Crawl the SIDC server from *start_year*/*start_month* to now and
        download everything available that is not already on disk.
        """
        now = utc_now()

        for channel in self.channels:
            print(f"\n[{channel}] Starting backfill ...")
            current = datetime(start_year, start_month, 1)

            while current <= now:
                year_str = current.strftime("%Y")
                month_str = current.strftime("%m")

                month_url = self._month_url(channel, current)
                try:
                    response = requests.get(month_url, timeout=30)
                    response.raise_for_status()
                except requests.RequestException:
                    print(f"  Skipping {year_str}/{month_str} - not available")
                    current = self._next_month(current)
                    continue

                soup = BeautifulSoup(response.text, "html.parser")
                day_folders = [
                    a["href"].strip("/")
                    for a in soup.find_all("a", href=True)
                    if a["href"].strip("/").isdigit()
                    and len(a["href"].strip("/")) == 2
                ]

                print(f"  {year_str}/{month_str} - found days: {sorted(day_folders)}")

                for day_str in sorted(day_folders):
                    day_url = f"{month_url}{day_str}/"
                    for filename in sorted(self._list_fits_in_url(day_url)):
                        try:
                            obs_time = self._parse_time_from_filename(filename)
                        except ValueError:
                            continue

                        save_dir = self._get_day_path(channel, obs_time)
                        fits_path = save_dir / filename
                        pickle_path = self._pickle_path(save_dir, obs_time)

                        self._download_file(day_url + filename, save_dir, filename)

                        if fits_path.exists() and not pickle_path.exists():
                            self._save_metadata(fits_path, save_dir)

                current = self._next_month(current)

    # ------------------------------------------------------------------
    # Download loop
    # ------------------------------------------------------------------

    def _download_channel_range(
        self, channel: str, start: datetime, end: datetime
    ) -> None:
        """Download every available frame for one channel within a window."""
        urls = self._list_files_in_range(channel, start, end)

        if not urls:
            print(f"[{channel}] No files found in range - skipping.")
            return

        for url, filename in urls:
            obs_time = self._parse_time_from_filename(filename)
            save_dir = self._get_day_path(channel, obs_time)
            self._download_file(url, save_dir, filename)

            fits_path = save_dir / filename
            pickle_path = self._pickle_path(save_dir, obs_time)

            if fits_path.exists() and not pickle_path.exists():
                self._save_metadata(fits_path, save_dir)

    # ------------------------------------------------------------------
    # URL / listing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sidc_folder(channel: str) -> str:
        """Return the SIDC wavelength folder name for a pipeline channel."""
        folder = _CHANNEL_TO_SIDC_FOLDER.get(channel)
        if folder is None:
            raise ValueError(
                f"Channel '{channel}' is not supported by the SIDC archive. "
                f"Supported channels: {list(_CHANNEL_TO_SIDC_FOLDER)}"
            )
        return folder

    def _month_url(self, channel: str, date: datetime) -> str:
        """Build the URL for a specific month directory on the SIDC server."""
        return (
            f"{_BASE_URL}/{self._sidc_folder(channel)}"
            f"/{date.strftime('%Y')}"
            f"/{date.strftime('%m')}/"
        )

    def _day_url(self, channel: str, date: datetime) -> str:
        """Build the URL for a specific day directory on the SIDC server."""
        return f"{self._month_url(channel, date)}{date.strftime('%d')}/"

    @staticmethod
    def _next_month(date: datetime) -> datetime:
        """Return the first day of the month following *date*."""
        if date.month == 12:
            return datetime(date.year + 1, 1, 1)
        return datetime(date.year, date.month + 1, 1)

    def _list_fits_in_url(self, url: str) -> List[str]:
        """
        Parse an Apache-style directory listing and return all .fits filenames.

        Returns an empty list when the page is unreachable or holds no FITS
        files, so a missing day does not stop the run.
        """
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"  Warning: could not reach {url} - {e}")
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        return [
            a["href"] for a in soup.find_all("a", href=True)
            if a["href"].endswith(".fits")
        ]

    def _list_files_in_range(
        self, channel: str, start: datetime, end: datetime
    ) -> List[tuple]:
        """
        Return ``(url, filename)`` pairs for all FITS files whose observation
        time falls within ``[start, end]``.
        """
        results = []
        current_day = start.replace(hour=0, minute=0, second=0, microsecond=0)

        while current_day <= end:
            day_url = self._day_url(channel, current_day)
            for filename in sorted(self._list_fits_in_url(day_url)):
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

        Raises
        ------
        ValueError
            If the filename does not match the expected pattern.
        """
        try:
            return datetime.strptime(filename.split(".")[2], "%Y%m%d_%H%M%S")
        except (IndexError, ValueError) as exc:
            raise ValueError(
                f"Cannot parse observation time from filename '{filename}'"
            ) from exc

    # ------------------------------------------------------------------
    # Download / metadata helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _download_file(url: str, save_dir: Path, filename: str) -> Optional[Path]:
        """
        Download a single FITS file, skipping it if it already exists.

        Returns
        -------
        Path or None
            Path to the downloaded file, or ``None`` if skipped or failed.
        """
        dest = save_dir / filename
        if dest.exists():
            print(f"  Already exists - skipping: {filename}")
            return None

        print(f"  Downloading: {filename} ...", end=" ", flush=True)
        started = time.time()
        try:
            response = requests.get(url, timeout=120, stream=True)
            response.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in response.iter_content(chunk_size=8192):
                    fh.write(chunk)
            size_kb = dest.stat().st_size / 1024
            print(f"done ({size_kb:.0f} KB, {time.time() - started:.1f}s)")
            return dest
        except requests.RequestException as e:
            print(f"FAILED - {e}")
            if dest.exists():
                dest.unlink()
            return None

    @staticmethod
    def _pickle_path(save_dir: Path, obs_time: datetime) -> Path:
        """Return the metadata pickle path for one observation time."""
        return save_dir / f"meta_data_{obs_time.strftime('%Y%m%d%H%M')}.pickle"

    def _save_metadata(self, filepath: Path, save_dir: Path) -> None:
        """
        Extract FITS header metadata and save it as a pickled DataFrame,
        following the same convention as the JSOC downloader.

        Also called when the FITS file exists but its pickle is missing, so
        metadata gaps are closed on a later run.
        """
        try:
            with fits.open(filepath) as hdul:
                header = hdul[1].header
                df = pd.DataFrame({key: [val] for key, val in header.items()})

            obs_time = self._parse_time_from_filename(filepath.name)
            pickle_path = self._pickle_path(save_dir, obs_time)
            self._write_pickle(df, pickle_path)
            print(f"  Metadata saved: {pickle_path.name}")
        except Exception as e:
            print(f"  Warning: could not save metadata for {filepath.name} - {e}")
