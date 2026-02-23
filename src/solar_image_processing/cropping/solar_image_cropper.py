import pickle
from copy import copy
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from joblib import Parallel, delayed, cpu_count
from skimage.measure import block_reduce
from skimage.transform import resize

from solar_image_processing.utils.pipeline_config import PipelineConfig
from solar_image_processing.utils.helper_functions import (
    find_missing_preprocessed_dates,
    find_missing_cropped_dates,
    load_existing_preprocessed_dates,
)


class ImageCropper:
    """
    Class for cropping and downsampling preprocessed solar images.

    Handles the full workflow: discovering images that need cropping,
    running parallel cropping, and validating completeness.

    Parameters
    ----------
    config : PipelineConfig
        Full pipeline configuration object. Used to extract channels,
        paths, date range, and cropping settings.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.channels = config.channels
        self.paths = config.paths
        self.start_date = config.start_date
        self.end_date = config.end_date
        self.config = config.cropping_config

        # Ensure output directories exist before processing starts
        self._create_output_directories(self.start_date, self.end_date)

    def run(self) -> None:
        """
        Crop all configured channels over the configured date range.
        """
        for channel in self.channels:
            print(f'\n{"="*60}')
            print(f'Cropping channel: {channel}')
            print(f'{"="*60}')

            self._process_channel(channel, self.start_date, self.end_date)

    def _create_output_directories(self, start: datetime, end: datetime) -> None:
        """
        Create the year/month directory tree for cropped output.

        Parameters
        ----------
        start : datetime
            Start of the date range.
        end : datetime
            End of the date range.
        """
        cropped_path = self.paths['cropped']
        cropped_path.mkdir(parents=True, exist_ok=True)

        for channel in self.channels:
            channel_path = cropped_path / channel
            channel_path.mkdir(exist_ok=True)

            current_month = copy(start)
            while current_month < end:
                year_path = channel_path / current_month.strftime('%Y')
                year_path.mkdir(exist_ok=True)

                month_path = year_path / current_month.strftime('%m')
                month_path.mkdir(exist_ok=True)

                current_month += relativedelta(months=1)

    def _process_channel(
        self,
        channel: str,
        start: datetime,
        end: datetime,
    ) -> None:
        """
        Crop all months for a single channel.

        Parameters
        ----------
        channel : str
            Channel identifier (e.g. ``'aia_171'``, ``'hmi'``).
        start : datetime
            Start of the date range.
        end : datetime
            End of the date range.
        """
        path_preprocessed = self.paths['preprocessed'] / channel
        path_cropped = Path(self.paths['cropped']) / channel

        current_month = datetime(start.year, start.month, 1)
        dates_to_check = []

        while current_month <= end:
            dates_to_check.extend(
                self._process_month(
                    channel, current_month, path_preprocessed, path_cropped
                )
            )
            current_month += relativedelta(months=1)

        print('\nAll cropping complete?')
        print(f'  {len(dates_to_check) == 0}')
        if dates_to_check:
            print(f'  Dates to check: {dates_to_check}')

    def _process_month(
        self,
        channel: str,
        month: datetime,
        path_preprocessed: Path,
        path_cropped: Path,
    ) -> List[datetime]:
        """
        Crop images for a single month.

        Parameters
        ----------
        channel : str
            Channel identifier.
        month : datetime
            Month to process (day component is ignored).
        path_preprocessed : Path
            Root preprocessed directory for this channel.
        path_cropped : Path
            Root cropped directory for this channel.

        Returns
        -------
        List[datetime]
            Dates that could not be cropped and need investigation.
        """
        print(f'\nProcessing {month.strftime("%Y/%m")}')

        path_input = path_preprocessed / month.strftime('%Y/%m')
        path_output = path_cropped / month.strftime('%Y/%m')

        existing_preprocessed = load_existing_preprocessed_dates(path_input, channel)
        missing_cropped, existing_cropped, _ = find_missing_cropped_dates(
            month, path_output, channel
        )

        # Only crop files that have not been cropped yet
        dates_to_crop = existing_preprocessed.difference(existing_cropped)

        if len(dates_to_crop) > 0:
            files_to_crop = self._build_file_list(channel, dates_to_crop)
            self._run_parallel_cropping(files_to_crop, path_input, path_output)

        # Re-check completeness after cropping
        missing_preprocessed, _ = find_missing_preprocessed_dates(
            month, path_input, channel
        )
        missing_cropped, _, _ = find_missing_cropped_dates(
            month, path_output, channel
        )

        if np.all(missing_preprocessed == missing_cropped):
            print(f'All images for {month.strftime("%Y/%m")} cropped successfully')
            return []
        else:
            print('Some preprocessed images were not cropped:')
            failed_dates = list(missing_cropped.difference(missing_preprocessed))
            return failed_dates

    def _build_file_list(self, channel: str, dates: pd.Index) -> List[str]:
        """
        Build the list of ``.npy`` filenames from a set of dates.

        Parameters
        ----------
        channel : str
            Channel identifier.
        dates : pd.Index
            Dates for which filenames are required.

        Returns
        -------
        List[str]
            Filenames in the format ``<channel_str>_<YYYY-MM-DD_HH:MM>.npy``.
        """
        # AIA channel strings are like 'aia_171'; extract '171' for filenames
        channel_str = channel.split('_')[1] if 'aia' in channel else channel
        return [
            f'{channel_str}_{date.strftime("%Y-%m-%d_%H:%M")}.npy'
            for date in dates
        ]

    def _run_parallel_cropping(
        self,
        files: List[str],
        path_input: Path,
        path_output: Path,
    ) -> None:
        """
        Crop a batch of files in parallel using half the available CPU cores.

        Parameters
        ----------
        files : List[str]
            Filenames to crop.
        path_input : Path
            Directory containing the input ``.npy`` files.
        path_output : Path
            Directory for cropped output files.
        """
        n_cpus = cpu_count()
        n_jobs = max(1, n_cpus // 2)
        print(f'Starting cropping with {n_jobs} workers')

        Parallel(n_jobs=n_jobs)(
            delayed(self._crop_single_image)(file, path_input, path_output)
            for file in files
        )

    def _crop_single_image(
        self,
        file: str,
        path_input: Path,
        path_output: Path,
    ) -> None:
        """
        Downsample, crop, and optionally resize a single preprocessed image.

        Parameters
        ----------
        file : str
            Filename of the ``.npy`` file to process.
        path_input : Path
            Directory containing preprocessed images.
        path_output : Path
            Directory for cropped output.
        """
        if not file.endswith('.npy'):
            return

        print(f'Processing image {file}')

        img = np.load(path_input / file)
        current_resolution = img.shape[0]

        if current_resolution % self.config['downsample_resolution'] != 0:
            raise ValueError(
                f'Image resolution {current_resolution} must be divisible by '
                f'target resolution {self.config["downsample_resolution"]}.'
            )

        # Sum-reduce blocks to preserve total flux during downsampling
        downsample_factor = current_resolution // self.config['downsample_resolution']
        img = block_reduce(img, (downsample_factor, downsample_factor), np.sum)

        img = self._apply_crop(img, file, path_input, downsample_factor)

        if self.config['resize_cropped'] is not None:
            img = resize(
                img,
                (self.config['resize_cropped'], self.config['resize_cropped']),
                order=3,
                mode='constant',
                cval=0,
            )

        # Save as float32 to reduce storage size
        np.save(path_output / file, img.astype('float32'))

    def _apply_crop(
        self,
        img: np.ndarray,
        file: str,
        path_input: Path,
        downsample_factor: int,
    ) -> np.ndarray:
        """
        Crop the image according to the configured crop mode.

        Supports two modes:

        - ``'square'``: symmetric pixel crop to a fixed size.
        - ``'disk'``: crop to the solar disk boundary using image metadata.

        Parameters
        ----------
        img : np.ndarray
            Downsampled image.
        file : str
            Original filename, used to locate the metadata pickle.
        path_input : Path
            Input directory containing metadata pickle files.
        downsample_factor : int
            Factor by which the image was downsampled, used to convert
            the disk radius from arcseconds to downsampled pixel units.

        Returns
        -------
        np.ndarray
            Cropped image.

        Raises
        ------
        ValueError
            If ``crop_mode`` is not ``'square'`` or ``'disk'``.
        """
        if self.config['crop_mode'] == 'square':
            # Symmetric crop to the configured pixel size
            cut_pixels = int((img.shape[0] - self.config['crop_pixels']) / 2)
            return img[cut_pixels:-cut_pixels, cut_pixels:-cut_pixels]

        elif self.config['crop_mode'] == 'disk':
            meta_file = path_input / file.replace('.npy', '_meta.pickle')
            with open(meta_file, 'rb') as f:
                meta_data = pickle.load(f)

            # Convert solar radius from arcseconds to downsampled pixel units
            downsampled_scale = meta_data['cdelt1'] * downsample_factor
            sun_radius_pixels = meta_data['rsun_obs'] / downsampled_scale
            cut_pixels = int(np.round((img.shape[0] - sun_radius_pixels * 2) / 2, 2))
            return img[cut_pixels:-cut_pixels, cut_pixels:-cut_pixels]

        else:
            raise ValueError('Crop mode not recognized.')
