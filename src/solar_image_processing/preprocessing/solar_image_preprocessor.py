from copy import copy
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
from dateutil.relativedelta import relativedelta
from joblib import Parallel, delayed, cpu_count

from solar_image_processing.preprocessing.aia_preprocessor import AIAPreprocessor
from solar_image_processing.preprocessing.hmi_preprocessor import HMIPreprocessor
from solar_image_processing.utils.pipeline_config import PipelineConfig
from solar_image_processing.utils.helper_functions import (
    find_missing_preprocessed_dates,
    load_existing_raw_files,
    load_config_data,
    find_files_to_preprocess,
    check_completeness_of_preprocessed_images,
)


class SolarImagePreprocessor:
    """
    Orchestrator for preprocessing SDO solar images.

    Manages the full preprocessing workflow: creating output directories,
    identifying missing preprocessed dates, running parallel preprocessing,
    and validating completeness.

    Parameters
    ----------
    config : PipelineConfig
        Full pipeline configuration object.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.channels = config.channels
        self.paths = config.paths
        self.start_date = config.start_date
        self.end_date = config.end_date
        self.config = config.preprocessing_config

        # Ensure output directories exist before processing starts
        self._create_output_directories()

    def run(self) -> None:
        """
        Preprocess all configured channels over the configured date range.
        """
        for channel in self.channels:
            print(f'\n{"="*60}')
            print(f'Processing channel: {channel}')
            print(f'{"="*60}')

            self._process_channel(channel)

    def _create_output_directories(self) -> None:
        """
        Create the year/month directory tree for preprocessed output.
        """
        preprocessed_path = Path(self.paths['preprocessed'])
        preprocessed_path.mkdir(parents=True, exist_ok=True)

        for channel in self.channels:
            channel_path = preprocessed_path / channel
            channel_path.mkdir(exist_ok=True)

            current_month = copy(self.start_date)
            while current_month < self.end_date:
                year_path = channel_path / current_month.strftime('%Y')
                year_path.mkdir(exist_ok=True)

                month_path = year_path / current_month.strftime('%m')
                month_path.mkdir(exist_ok=True)

                current_month += relativedelta(months=1)

    def _process_channel(self, channel: str) -> None:
        """
        Preprocess all months for a single channel.

        Parameters
        ----------
        channel : str
            Channel identifier (e.g. ``'aia_171'``, ``'hmi'``).
        """
        # Raw FITS files are stored in instrument-specific subdirectories
        if channel[:3] == 'aia':
            path_raw = self.paths['unprocessed'] / 'AIA' / channel[-3:]
        elif channel[:3] == 'hmi':
            path_raw = self.paths['unprocessed'] / 'HMI' / 'magnetogram'

        path_preprocessed = Path(self.paths['preprocessed']) / channel
        current_month = datetime(self.start_date.year, self.start_date.month, 1)
        dates_to_check = []

        while current_month <= self.end_date:
            dates_to_check.extend(
                self._process_month(channel, current_month, path_raw, path_preprocessed)
            )
            current_month += relativedelta(months=1)

        print('\nAll preprocessing complete?')
        print(f'  {len(dates_to_check) == 0}')
        if dates_to_check:
            print(f'  Dates to check: {dates_to_check}')

    def _process_month(
        self,
        channel: str,
        month: datetime,
        path_raw: Path,
        path_preprocessed: Path,
    ) -> List[datetime]:
        """
        Preprocess all images for a single month.

        Parameters
        ----------
        channel : str
            Channel identifier.
        month : datetime
            Month to process (day component is ignored).
        path_raw : Path
            Root raw FITS directory for this channel.
        path_preprocessed : Path
            Root preprocessed output directory for this channel.

        Returns
        -------
        List[datetime]
            Dates that could not be preprocessed and need investigation.
        """
        print(f'\nProcessing {month.strftime("%Y/%m")}')

        path_raw_month = path_raw / month.strftime('%Y/%m')
        path_output = path_preprocessed / month.strftime('%Y/%m')

        existing_raw_files = load_existing_raw_files(path_raw_month)
        missing_dates, _ = find_missing_preprocessed_dates(
            month, path_output, channel, self.config['overwrite_existing']
        )

        # Optionally exclude dates that previously failed preprocessing
        files_to_exclude = self._load_exclusion_list(path_output)
        if self.config['load_preprocessing_fails'] and not files_to_exclude.empty:
            missing_dates = missing_dates.difference(files_to_exclude.index)

        if len(missing_dates) > 0:
            files_to_preprocess, new_exclusions = find_files_to_preprocess(
                missing_dates, existing_raw_files, path_raw_month
            )

            # Merge with any previous exclusions before saving
            if not files_to_exclude.empty:
                new_exclusions = pd.concat([files_to_exclude, new_exclusions])
            new_exclusions.to_csv(path_output / 'preprocessing_fails.csv')
        else:
            files_to_preprocess = pd.Series(dtype=object)
            new_exclusions = files_to_exclude

        if len(files_to_preprocess) > 0:
            self._run_parallel_preprocessing(
                channel, files_to_preprocess, path_raw_month, path_output, month
            )

        _, dates_to_check = check_completeness_of_preprocessed_images(
            new_exclusions, month, path_output, channel
        )

        return dates_to_check

    def _load_exclusion_list(self, path_output: Path) -> pd.DataFrame:
        """
        Load the CSV of previously failed preprocessing dates.

        Returns an empty DataFrame if ``load_preprocessing_fails`` is
        disabled in the configuration or if no file exists yet.

        Parameters
        ----------
        path_output : Path
            Directory that may contain ``preprocessing_fails.csv``.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``['bad', 'missing_raw']``.
        """
        if not self.config['load_preprocessing_fails']:
            return pd.DataFrame(columns=['bad', 'missing_raw'])

        try:
            return pd.read_csv(
                path_output / 'preprocessing_fails.csv',
                index_col=0,
                parse_dates=[0],
            )
        except FileNotFoundError:
            return pd.DataFrame(columns=['bad', 'missing_raw'])

    def _run_parallel_preprocessing(
        self,
        channel: str,
        files_to_preprocess: pd.Series,
        path_input: Path,
        path_output: Path,
        month: datetime,
    ) -> None:
        """
        Preprocess a batch of files in parallel using half the available CPU cores.

        Parameters
        ----------
        channel : str
            Channel identifier; determines which preprocessor class is used.
        files_to_preprocess : pd.Series
            Mapping from FITS filenames to target dates.
        path_input : Path
            Directory containing raw FITS files.
        path_output : Path
            Directory for preprocessed output files.
        month : datetime
            Month being processed; used to load AIA calibration data.
        """
        n_cpus = cpu_count()
        n_jobs = max(1, n_cpus // 2)
        print(f'Starting preprocessing with {n_jobs} workers')

        # Each file may map to multiple target dates; deduplicate for iteration
        unique_files = files_to_preprocess.index.unique()

        if channel[:3] == 'hmi':
            preprocessor = HMIPreprocessor(self.config)
            Parallel(n_jobs=n_jobs)(
                delayed(preprocessor.process_file)(
                    file, files_to_preprocess, path_input, path_output
                )
                for file in unique_files
            )
        elif channel[:3] == 'aia':
            # PSF, degradation correction, and pointing tables are month-specific
            psf, correction_table, pointing_table = load_config_data(
                self.paths['instrument_data'], channel[-3:], month
            )
            preprocessor = AIAPreprocessor(pointing_table, psf, correction_table, self.config)
            Parallel(n_jobs=n_jobs)(
                delayed(preprocessor.process_file)(
                    file, files_to_preprocess, path_input, path_output
                )
                for file in unique_files
            )
