"""
Solar Image Downloader Module.

This module provides functionality to download SDO (Solar Dynamics Observatory)
solar images from the JSOC (Joint Science Operations Center) archive. It supports
downloading images at hourly cadence for both AIA (Atmospheric Imaging Assembly)
and HMI (Helioseismic and Magnetic Imager) instruments.

Key Features:
- Download images at hourly cadence
- Automatic fallback to adjacent images when primary images are unavailable
- Quality checking and substitute date tracking
- Support for AIA EUV channels (171, 193, 211 Angstrom) and HMI magnetograms

Example Usage:
    config = DownloadConfiguration(
        client_email='user@example.com',
        series='AIA.lev1_euv_12s',
        wavelength=171,
        segment='image',
        rebin_factor=4
    )
    downloader = SolarImageDownloader(config, path_downloaded, path_preprocessed)
    downloader.download_images_hourly_cadence(start_date)
"""

from __future__ import annotations

import os
import pickle
import time
from copy import copy
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from solar_image_processing.downloading import jsoc_download as jd
from solar_image_processing.utils import check_file_quality, find_missing_preprocessed_dates, read_file_name

# import donwload configuration from solar_image_downloader
@dataclass
class DownloadConfiguration:
    """
    Configuration settings for solar image downloads.

    Parameters
    ----------
    client_email : str
        Email address registered with JSOC for data requests.
        Must be registered at http://jsoc.stanford.edu/ajax/exportdata.html
    series : str
        JSOC data series name (e.g., 'AIA.lev1_euv_12s' or 'hmi.M_720s').
    wavelength : Union[int, str]
        Wavelength for AIA images (e.g., 171, 193, 211) or empty string for HMI.
    segment : str
        Data segment type ('image' for AIA, 'magnetogram' for HMI).
    rebin_factor : int, optional
        Factor to rebin images on server before download.
        Default is 4 (results in 1024x1024 images from 4096x4096 originals).
    """

    client_email: str
    series: str
    wavelength: Union[int, str]
    segment: str
    rebin_factor: int = 4

    @property
    def is_aia(self) -> bool:
        """Check if the configuration is for AIA data."""
        return self.series[:3] == 'AIA'

    @property
    def is_hmi(self) -> bool:
        """Check if the configuration is for HMI data."""
        return self.series[:3] == 'hmi'

    @property
    def base_cadence_minutes(self) -> int:
        """
        Get the base cadence in minutes for the instrument.

        AIA EUV images have 12-second cadence, but for substitute searching
        we use 1-minute intervals. HMI magnetograms have 12-minute cadence.
        """
        if self.is_aia:
            return 1 # minute
        elif self.is_hmi:
            return 12 # minutes
        else:
            raise ValueError(f"Unknown instrument type: {self.series[:3]}")


class SolarImageSubstituteFinder:
    """
    Main class for downloading SDO solar images from JSOC.

    This class handles downloading solar images at hourly cadence,
    with automatic fallback to adjacent images when primary images are
    unavailable. It supports both AIA EUV images and HMI magnetograms.

    Parameters
    ----------
    config : DownloadConfiguration
        Configuration settings for downloads.
    path_downloaded : Union[str, Path]
        Base path for storing downloaded FITS files.
    path_preprocessed : Union[str, Path]
        Base path for preprocessed data (used to check for missing images).

    Examples
    --------
    >>> config = DownloadConfiguration(
    ...     client_email='user@example.com',
    ...     series='AIA.lev1_euv_12s',
    ...     wavelength=171,
    ...     segment='image'
    ... )
    >>> downloader = SolarImageDownloader(config, '/data/downloaded/', '/data/preprocessed/')
    >>> downloader.download_images_hourly_cadence(datetime(2020, 1, 1))
    """

    # Maximum time offset (in minutes) to search for substitute images
    MAX_SUBSTITUTE_OFFSET_MINUTES: int = 48

    def __init__(
        self,
        config: DownloadConfiguration,
        path_downloaded: Union[str, Path],
        path_preprocessed: Union[str, Path]
    ) -> None:
        self.config = config
        self.path_downloaded = Path(path_downloaded)
        self.path_preprocessed = Path(path_preprocessed)

        # Ensure base directories exist
        self.path_downloaded.mkdir(parents=True, exist_ok=True)
        self.path_preprocessed.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Path Helper Methods
    # -------------------------------------------------------------------------

    def _get_month_path(self, date: datetime, create: bool = True) -> Path:
        """Get the path for a specific month's downloaded data."""
        year_path = self.path_downloaded / str(date.year)
        month_path = year_path / f"{date.month:02d}"

        if create:
            year_path.mkdir(exist_ok=True)
            month_path.mkdir(exist_ok=True)

        return month_path

    # -------------------------------------------------------------------------
    # Primary Download Methods
    # -------------------------------------------------------------------------

    def download_images_hourly_cadence(self, start_date: datetime) -> None:
        """
        Download images at hourly cadence starting from the given date.

        Downloads one day of images at a time until reaching today's date.
        If a daily batch download fails, falls back to downloading
        individual hours.

        Parameters
        ----------
        start_date : datetime
            Date to start downloading from.
        """
        current_date = start_date

        while current_date < datetime.today():
            start_time = time.time()
            print(f"Requesting images for {current_date}")

            end_date = current_date + timedelta(hours=24)
            month_path = self._get_month_path(current_date)

            # Attempt batch download for the entire day
            success = self._download_daily_batch(current_date, end_date, month_path)

            if not success:
                # Fall back to downloading individual hours
                self._download_hourly_fallback(current_date, month_path)

            elapsed = time.time() - start_time
            print(f"--- {elapsed:.2f} seconds ---")

            current_date = copy(end_date)

    def _download_daily_batch(
        self,
        start_date: datetime,
        end_date: datetime,
        month_path: Path
    ) -> bool:
        """Attempt to download a full day's worth of images in one batch."""
        client = jd.client(self.config.client_email)
        request_string = client.create_request_string(
            self.config.series,
            start_date,
            endtime=end_date,
            wavelength=self.config.wavelength,
            segment=self.config.segment,
            period='',
            cadence=timedelta(hours=1)
        )

        search_results = client.search(request_string, keys=['t_obs', '**ALL**'])
        print("Request successful. Start downloading.")

        # Save metadata for the day
        self._save_metadata(search_results, month_path, start_date, '%Y%m%d')

        try:
            client.download(
                request_string,
                str(month_path),
                method='url-tar',
                protocol='fits',
                filter=None,
                rebin=self.config.rebin_factor,
                process={}
            )
            print("Files downloaded successfully.")
            return True
        except Exception as e:
            print(f"File download error for {start_date}: {e}")
            print("Trying hourly downloads as fallback.")
            return False

    def _download_hourly_fallback(self, start_date: datetime, month_path: Path) -> None:
        """Download images one hour at a time as a fallback strategy."""
        current_hour = copy(start_date)

        for i in range(24):
            print(f"Requesting single hour: {current_hour}")
            end_hour = current_hour + timedelta(hours=1)

            client = jd.client(self.config.client_email)
            request_string = client.create_request_string(
                self.config.series,
                current_hour,
                endtime=end_hour,
                wavelength=self.config.wavelength,
                segment=self.config.segment,
                period='',
                cadence=timedelta(hours=1)
            )

            search_results = client.search(request_string, keys=['t_obs', '**ALL**'])
            #print(search_results)
            print("Request for single hour successful. Start downloading.")

            self._save_metadata(search_results, month_path, current_hour, '%Y%m%d%H')

            try:
                client.download(
                    request_string,
                    str(month_path),
                    method='url-tar',
                    protocol='fits',
                    filter=None,
                    rebin=self.config.rebin_factor,
                    process={}
                )
                print("Single file downloaded successfully.")
            except Exception as e:
                print(f"Single file download error for {current_hour}: {e}")
                print("Skipping this hour.")

            current_hour = current_hour + timedelta(hours=1)

    # -------------------------------------------------------------------------
    # Adjacent Image Download Methods
    # -------------------------------------------------------------------------

    def download_adjacent_images(
        self,
        date: datetime,
        month_path: Path,
        interval_minutes: int
    ) -> list[str]:
        """
        Download images adjacent to a missing timestamp.

        When the primary hourly image is unavailable, this method downloads
        images from nearby times that can serve as substitutes.

        Parameters
        ----------
        date : datetime
            The target timestamp that is missing an image.
        month_path : Path
            Directory to save downloaded files.
        interval_minutes : int
            Time interval (in minutes) before and after the target date
            to search for images.

        Returns
        -------
        list[str]
            List of downloaded filenames, or empty list if download failed.
        """
        # Calculate search window based on instrument cadence
        # AIA has faster cadence, so we add a smaller buffer
        # HMI needs a larger buffer due to its 12-minute cadence
        start_date = date - timedelta(minutes=interval_minutes)
        if self.config.is_aia:
            end_date = date + timedelta(minutes=interval_minutes + 5)
        elif self.config.is_hmi:
            end_date = date + timedelta(minutes=interval_minutes + 12)
        else:
            end_date = date + timedelta(minutes=interval_minutes + 5)

        cadence = timedelta(minutes=interval_minutes * 2)
        client = jd.client(self.config.client_email)

        try:
            request_string = client.create_request_string(
                self.config.series,
                start_date,
                endtime=end_date,
                wavelength=self.config.wavelength,
                segment=self.config.segment,
                period='',
                cadence=cadence
            )
            search_results = client.search(request_string, keys=['t_obs', '**ALL**'])
            print(search_results)
        except Exception as e:
            print(f"Failed to request {start_date} to {end_date} with {cadence} cadence: {e}")
            return []

        if search_results.empty:
            return []

        self._save_metadata(search_results, month_path, start_date, '%Y%m%d%H%M')

        try:
            files_downloaded = client.download(
                request_string,
                str(month_path),
                method='url-tar',
                protocol='fits',
                filter=None,
                rebin=self.config.rebin_factor,
                process={}
            )
            print(files_downloaded)
            return files_downloaded
        except Exception as e:
            print(f"Failed to download {start_date} to {end_date}: {e}")
            return []

    # -------------------------------------------------------------------------
    # Missing Image Detection and Handling
    # -------------------------------------------------------------------------

    def download_missing_images(self, start_date: datetime) -> None:
        """
        Check for and download missing images for each month from start_date.

        Iterates through each month from start_date to today, checking for
        missing images and attempting to download substitutes.

        Parameters
        ----------
        start_date : datetime
            Date to start checking from.
        """
        current_date = start_date

        while current_date < datetime.today():
            month = datetime(current_date.year, current_date.month, 1, 0)
            self.check_for_missing_images(month)
            current_date = current_date + relativedelta(months=1)

    def check_for_missing_images(self, month: datetime) -> None:
        """
        Check for missing images in a month and download substitutes.

        This method:
        1. Identifies target hourly timestamps for the month
        2. Finds which images are missing (not downloaded or failed preprocessing)
        3. Attempts to find or download substitute images
        4. Saves the mapping of missing dates to substitute dates

        Parameters
        ----------
        month : datetime
            The month to check (day/hour components are ignored).
        """
        month_path = self._get_month_path(month, create=False)
        preprocessed_path = self.path_preprocessed / month.strftime('%Y/%m')

        # Determine the date range for this month (SDO data begins May 13, 2010)
        if month.year == 2010 and month.month == 5:
            month_start = datetime(2010, 5, 13, 0)
            month_end = datetime(2010, 5, 31, 23)
        else:
            month_start = datetime(month.year, month.month, 1, 0)
            month_end = month_start + relativedelta(months=1) - timedelta(hours=1)

        # Find existing downloaded files and their timestamps
        existing_hourly, existing_small_cadence = self._scan_existing_files(month_path)

        # Identify missing timestamps
        missing_dates = self._identify_missing_dates(
            month_start, month_end, existing_hourly, preprocessed_path
        )
        print(missing_dates)

        # Process each missing date to find or download substitutes
        substitute_dates = pd.Series(
            np.zeros(len(missing_dates), dtype=object),
            index=missing_dates
        )
        for date in missing_dates:
            substitute_dates.loc[date] = self._find_substitute_for_date(
                date, existing_small_cadence, month_path
            )

        # Save the substitute dates mapping
        substitute_dates.to_csv(month_path / 'substitute_dates.csv')

    def _scan_existing_files(self, month_path: Path) -> tuple[pd.Index, pd.Series]:
        """Scan downloaded files and extract their timestamps."""
        files = sorted(os.listdir(month_path))
        hourly_dates: list[datetime] = []
        small_cadence_files: list[str] = []
        small_cadence_dates: list[datetime] = []

        cadence_minutes = self.config.base_cadence_minutes

        for file in files:
            if not file.endswith('.fits'):
                continue

            file_date, _, _ = read_file_name(file)

            # Round to hourly for primary matching
            hourly_dates.append(
                pd.Timestamp(file_date).round('60min').to_pydatetime()
            )

            # Also track at finer cadence for substitute searching
            small_cadence_files.append(file)
            small_cadence_dates.append(
                pd.Timestamp(file_date).round(f'{cadence_minutes}min').to_pydatetime()
            )

        return pd.Index(hourly_dates), pd.Series(small_cadence_dates, index=small_cadence_files)

    def _identify_missing_dates(
        self,
        month_start: datetime,
        month_end: datetime,
        existing_hourly: pd.Index,
        preprocessed_path: Path
    ) -> pd.Index:
        """Identify which hourly timestamps are missing data."""
        # Generate target hourly timestamps for the full month
        target_dates = pd.date_range(month_start, month_end, freq='1h')

        # Find dates missing from downloads
        missing_downloaded = target_dates.difference(existing_hourly)

        # Find dates that failed preprocessing
        month = datetime(month_start.year, month_start.month, 1)
        missing_preprocessed, _ = find_missing_preprocessed_dates(
            month, str(preprocessed_path), self.config.wavelength
        )

        # Union of both missing sets
        return missing_downloaded.union(missing_preprocessed)

    def _find_substitute_for_date(
        self,
        date: datetime,
        existing_small_cadence: pd.Series,
        month_path: Path
    ) -> Optional[datetime]:
        """Find a substitute image for a single missing date."""
        # Check if there are already candidate images nearby
        good_dates, downloaded_all = self._check_existing_candidates(
            date, existing_small_cadence, month_path
        )

        if good_dates:
            return self._select_best_substitute(date, good_dates)

        # If we haven't downloaded all possible candidates, try downloading
        if not downloaded_all:
            good_dates = self._download_and_check_candidates(date, month_path)
            if good_dates:
                return self._select_best_substitute(date, good_dates)

        print(f"No substitute date for {date} found.")
        return None

    def _check_existing_candidates(
        self,
        date: datetime,
        existing_small_cadence: pd.Series,
        month_path: Path
    ) -> tuple[list[datetime], bool]:
        """Check existing downloaded files for valid substitute candidates."""
        # Find files within +/- 49 minutes of target
        mask = (
            (existing_small_cadence.values > date - timedelta(minutes=49)) &
            (existing_small_cadence.values < date + timedelta(minutes=49))
        )

        if np.sum(mask) <= 1:
            print(f"Missing image: {date}, no substitute candidates found.")
            return [], False

        print(f"Missing image: {date}, substitute candidates found.")
        candidate_files = existing_small_cadence.index[mask]
        good_dates, _ = check_file_quality(candidate_files, month_path)

        # Check if we've already downloaded all candidates within range
        existing_dates = np.sort(existing_small_cadence.loc[mask].values)
        if self.config.is_aia:
            earliest_needed = date - timedelta(minutes=45)
            latest_needed = date + timedelta(minutes=45)
        elif self.config.is_hmi:
            earliest_needed = date - timedelta(minutes=48)
            latest_needed = date + timedelta(minutes=48)
        else:
            return good_dates, False

        downloaded_all = (earliest_needed in existing_dates and latest_needed in existing_dates)
        return good_dates, downloaded_all

    def _download_and_check_candidates(
        self,
        date: datetime,
        month_path: Path
    ) -> list[datetime]:
        """Download adjacent images and check their quality."""
        print(f"Missing image: {date}, downloading adjacent images.")

        interval = self.config.base_cadence_minutes
        good_dates: list[datetime] = []

        while interval <= self.MAX_SUBSTITUTE_OFFSET_MINUTES:
            print(f"Downloading images {interval} minutes around {date}")

            downloaded_files = self.download_adjacent_images(date, month_path, interval)
            good_dates, bad_dates = check_file_quality(downloaded_files, month_path)
            print(f"Good: {len(good_dates)}, Bad: {len(bad_dates)}")

            if good_dates:
                return good_dates

            interval += self.config.base_cadence_minutes

        return good_dates

    def _select_best_substitute(
        self,
        date: datetime,
        good_dates: list[datetime]
    ) -> datetime:
        """Select the best substitute (closest in time) from valid candidates."""
        time_differences = np.abs(date - pd.Index(good_dates))
        best_substitute = good_dates[np.argmin(time_differences)]
        print(f"Substitute date for {date} successfully determined: {best_substitute}")
        return best_substitute

    # -------------------------------------------------------------------------
    # Substitute Date Collection
    # -------------------------------------------------------------------------

    def collect_all_substitute_dates(
        self,
        start_month: datetime,
        end_month: datetime
    ) -> pd.Series:
        """
        Collect substitute dates from all months into a single file.

        Iterates through all months in the date range and merges
        individual substitute_dates.csv files.

        Parameters
        ----------
        start_month : datetime
            First month to include.
        end_month : datetime
            Last month to include.

        Returns
        -------
        pd.Series
            Merged series of all substitute dates.
        """
        current_month = copy(start_month)
        collected_data: list[pd.Series] = []

        while current_month <= end_month:
            csv_path = (
                self.path_downloaded
                / current_month.strftime('%Y/%m')
                / 'substitute_dates.csv'
            )
            try:
                month_data = pd.read_csv(
                    csv_path, index_col=0, parse_dates=[0, 1]
                ).squeeze()
                collected_data.append(month_data)
            except FileNotFoundError:
                print(f"{current_month.strftime('%Y/%m')} does not exist!")
            except Exception as e:
                print(f"Error reading {current_month.strftime('%Y/%m')}: {e}")

            current_month = current_month + relativedelta(months=1)

        if not collected_data:
            raise ValueError("No substitute date files found in the date range.")

        substitute_dates = pd.concat(collected_data, axis=0)

        # Save merged file
        substitute_dates.to_csv(self.path_downloaded / 'substitute_dates_merged.csv')
        print(substitute_dates)

        return substitute_dates

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def _save_metadata(
        self,
        search_results: pd.DataFrame,
        month_path: Path,
        date: datetime,
        date_format: str
    ) -> None:
        """Save search results metadata to a pickle file."""
        filepath = month_path / f"meta_data_{date.strftime(date_format)}.pickle"
        with open(filepath, 'wb') as f:
            pickle.dump(search_results, f)


# =============================================================================
# Convenience Functions
# =============================================================================


def create_aia_config(
    email: str,
    wavelength: int,
    rebin_factor: int = 4
) -> DownloadConfiguration:
    """
    Create a download configuration for AIA EUV images.

    Parameters
    ----------
    email : str
        JSOC-registered email address.
    wavelength : int
        AIA wavelength in Angstroms (e.g., 171, 193, 211).
    rebin_factor : int, optional
        Rebinning factor for image size reduction. Default is 4.
    """
    return DownloadConfiguration(
        client_email=email,
        series='AIA.lev1_euv_12s',
        wavelength=wavelength,
        segment='image',
        rebin_factor=rebin_factor
    )


def create_hmi_config(email: str, rebin_factor: int = 4) -> DownloadConfiguration:
    """
    Create a download configuration for HMI magnetograms.

    Parameters
    ----------
    email : str
        JSOC-registered email address.
    rebin_factor : int, optional
        Rebinning factor for image size reduction. Default is 4.
    """
    return DownloadConfiguration(
        client_email=email,
        series='hmi.M_720s',
        wavelength='',
        segment='magnetogram',
        rebin_factor=rebin_factor
    )


def get_data_paths(
    base_dir: Union[str, Path],
    config: DownloadConfiguration
) -> tuple[Path, Path]:
    """
    Get the appropriate data paths based on configuration.

    Parameters
    ----------
    base_dir : Union[str, Path]
        Base directory for data storage.
    config : DownloadConfiguration
        Download configuration to determine instrument type.

    Returns
    -------
    tuple[Path, Path]
        Tuple of (path_downloaded, path_preprocessed).
    """
    base_dir = Path(base_dir)

    if config.is_aia:
        path_downloaded = (
            base_dir / 'data' / 'unprocessed_data' / 'SDO' / 'AIA' / str(config.wavelength)
        )
        path_preprocessed = (
            base_dir / 'data' / 'processed_data' / 'deep_learning'
            / f'aia_{config.wavelength}'
        )
    else:
        path_downloaded = (
            base_dir / 'data' / 'unprocessed_data' / 'SDO' / 'HMI' / 'magnetogram'
        )
        path_preprocessed = (
            base_dir / 'data' / 'processed_data' / 'deep_learning' / 'hmi'
        )

    return path_downloaded, path_preprocessed
