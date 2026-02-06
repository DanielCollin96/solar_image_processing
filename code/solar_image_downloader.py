"""
Solar Image Downloader Module.

This module provides functionality to download SDO (Solar Dynamics Observatory)
solar images from the JSOC (Joint Science Operations Center) archive. It supports
downloading images at hourly cadence for both AIA (Atmospheric Imaging Assembly)
and HMI (Helioseismic and Magnetic Imager) instruments.

Key Features:
- Download images at hourly cadence
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

#from __future__ import annotations
import pickle
import time
from copy import copy
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Union
import pandas as pd

import jsoc_download as jd


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


class SolarImageDownloader:
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
