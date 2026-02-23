import pickle
import time
from copy import copy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Union

import pandas as pd
from dateutil.relativedelta import relativedelta

from solar_image_processing.downloading import jsoc_download as jd
from solar_image_processing.utils.pipeline_config import PipelineConfig


class SolarImageDownloader:
    """
    Download SDO solar images from JSOC at hourly cadence.

    Supports AIA EUV images and HMI magnetograms. For each day, a batch
    download is attempted first; individual hourly downloads are used as
    a fallback if the batch fails.

    Parameters
    ----------
    config : PipelineConfig
        Full pipeline configuration object. Used to extract channels,
        paths, date range, and download settings.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config.download_config
        self.channels = config.channels
        self.start_date = config.start_date
        self.end_date = config.end_date

        self.path_downloaded = config.paths['unprocessed']
        self._create_download_directories()

    def _create_download_directories(self) -> None:
        """
        Create the year/month directory tree for downloaded FITS files.

        AIA images are stored under ``AIA/<wavelength>/``;
        HMI images under ``HMI/magnetogram/``.
        """
        self.path_downloaded.mkdir(parents=True, exist_ok=True)

        for channel in self.channels:
            if channel[:3] == 'aia':
                channel_path = self.path_downloaded / 'AIA' / channel[-3:]
            elif channel[:3] == 'hmi':
                channel_path = self.path_downloaded / 'HMI' / 'magnetogram'

            channel_path.mkdir(parents=True, exist_ok=True)

            current_month = copy(self.start_date)
            while current_month < self.end_date:
                year_path = channel_path / current_month.strftime('%Y')
                year_path.mkdir(exist_ok=True)

                month_path = year_path / current_month.strftime('%m')
                month_path.mkdir(exist_ok=True)

                current_month += relativedelta(months=1)

    def _get_month_path(
        self,
        date: datetime,
        channel: str,
        create: bool = True,
    ) -> Path:
        """
        Return the storage path for a given month and channel.

        Parameters
        ----------
        date : datetime
            Date whose year/month subdirectory is needed.
        channel : str
            Channel identifier (e.g. ``'aia_171'``, ``'hmi'``).
        create : bool, optional
            If ``True``, create the directory if it does not exist.
            Default is ``True``.

        Returns
        -------
        Path
            Path to the ``<year>/<month>`` directory for that channel.
        """
        if channel[:3] == 'aia':
            channel_path = self.path_downloaded / 'AIA' / channel[-3:]
        elif channel[:3] == 'hmi':
            channel_path = self.path_downloaded / 'HMI' / 'magnetogram'

        year_path = channel_path / date.strftime('%Y')
        month_path = year_path / date.strftime('%m')

        if create:
            year_path.mkdir(exist_ok=True)
            month_path.mkdir(exist_ok=True)

        return month_path

    def download_images_hourly_cadence(self) -> None:
        """
        Download images at hourly cadence for all configured channels.

        Iterates day by day from ``start_date`` to ``end_date``. For each
        day, a single batch download is attempted first. If it fails, the
        method falls back to downloading each hour individually.
        """
        for channel in self.channels:
            current_date = self.start_date

            if self.end_date is None:
                self.end_date = datetime.today()

            while current_date < self.end_date:
                start_time = time.time()
                print(f"Requesting images for {current_date}")

                daily_batch_end_date = min(
                    current_date + timedelta(hours=24), self.end_date
                )
                month_path = self._get_month_path(current_date, channel)

                # Attempt batch download for the entire day
                success = self._download_daily_batch(
                    current_date, daily_batch_end_date, month_path, channel
                )

                if not success:
                    # Fall back to downloading individual hours
                    self._download_hourly_fallback(current_date, month_path, channel)

                elapsed = time.time() - start_time
                print(f"--- {elapsed:.2f} seconds ---")

                current_date = copy(daily_batch_end_date)

    def _download_daily_batch(
        self,
        start_date: datetime,
        end_date: datetime,
        month_path: Path,
        channel: str,
    ) -> bool:
        """
        Attempt to download a full day of images in a single batch request.

        Parameters
        ----------
        start_date : datetime
            Start of the download window (inclusive).
        end_date : datetime
            End of the download window (exclusive).
        month_path : Path
            Directory where downloaded files are saved.
        channel : str
            Channel identifier (e.g. ``'aia_171'``, ``'hmi'``).

        Returns
        -------
        bool
            ``True`` if the download succeeded, ``False`` otherwise.
        """
        client = jd.client(self.config['email'])

        # AIA channels encode the wavelength; HMI has no wavelength field
        wavelength = channel[-3:] if channel[:3] == 'aia' else ''

        request_string = client.create_request_string(
            self.config['jsoc_series'][channel[:3]]['series'],
            start_date,
            endtime=end_date,
            wavelength=wavelength,
            segment=self.config['jsoc_series'][channel[:3]]['segment'],
            period='',
            cadence=timedelta(hours=1),
        )
        print(request_string)
        search_results = client.search(request_string, keys=['t_obs', '**ALL**'])
        print("Request successful. Meta data:")
        print(search_results)
        print("Start downloading.")

        self._save_metadata(search_results, month_path, start_date, '%Y%m%d')

        try:
            client.download(
                request_string,
                str(month_path),
                method='url-tar',
                protocol='fits',
                filter=None,
                rebin=self.config['rebin_factor'],
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
        month_path: Path,
        channel: str,
    ) -> None:
        """
        Download images one hour at a time as a fallback strategy.

        Called when the daily batch download fails. Each hour is requested
        independently; failed hours are skipped with a printed warning.

        Parameters
        ----------
        start_date : datetime
            Start of the day to download (00:00 of that day).
        month_path : Path
            Directory where downloaded files are saved.
        channel : str
            Channel identifier (e.g. ``'aia_171'``, ``'hmi'``).
        """
        current_hour = copy(start_date)

        for i in range(24):
            print(f"Requesting single hour: {current_hour}")
            end_hour = current_hour + timedelta(hours=1)

            client = jd.client(self.config['email'])

            # AIA channels encode the wavelength; HMI has no wavelength field
            wavelength = channel[-3:] if channel[:3] == 'aia' else ''

            request_string = client.create_request_string(
                self.config['jsoc_series'][channel[:3]]['series'],
                current_hour,
                endtime=end_hour,
                wavelength=wavelength,
                segment=self.config['jsoc_series'][channel[:3]]['segment'],
                period='',
                cadence=timedelta(hours=1),
            )

            search_results = client.search(request_string, keys=['t_obs', '**ALL**'])
            print("Request for single hour successful. Start downloading.")

            self._save_metadata(search_results, month_path, current_hour, '%Y%m%d%H')

            try:
                client.download(
                    request_string,
                    str(month_path),
                    method='url-tar',
                    protocol='fits',
                    filter=None,
                    rebin=self.config['rebin_factor'],
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
        month_path: Path,
        date: datetime,
        date_format: str,
    ) -> None:
        """
        Save JSOC search result metadata to a pickle file.

        Parameters
        ----------
        search_results : pd.DataFrame
            Metadata returned by the JSOC search query.
        month_path : Path
            Directory where the pickle file is saved.
        date : datetime
            Date used to construct the filename.
        date_format : str
            ``strftime`` format string applied to ``date`` for the filename
            (e.g. ``'%Y%m%d'`` for daily, ``'%Y%m%d%H'`` for hourly).
        """
        filepath = month_path / f"meta_data_{date.strftime(date_format)}.pickle"
        with open(filepath, 'wb') as f:
            pickle.dump(search_results, f)
