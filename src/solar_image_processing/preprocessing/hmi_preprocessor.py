from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple

import astropy.units as u
import numpy as np
import pandas as pd
import sunpy.map
from sunpy.map import contains_full_disk

from solar_image_processing.utils.helper_functions import (
    read_file_name,
    save_preprocessed_output,
)
from solar_image_processing.preprocessing.preprocessing_functions import (
    register_image,
    scale_solar_disk_radius,
    compute_differential_rotation,
)


class HMIPreprocessor:
    """
    Preprocessor for HMI (Helioseismic and Magnetic Imager) magnetograms.

    Implements the HMI preprocessing pipeline:

    1. Upsample to 4096 px.
    2. Apply differential rotation correction if needed.
    3. Downsample to 1024 px.
    4. Register (align) with zero fill for off-disk regions.
    5. Replace NaN values with zero.
    6. Normalise solar disk radius.
    7. Flip to solar north-up orientation.

    Parameters
    ----------
    config : dict
        Preprocessing configuration. Expected keys:
        ``'target_rsun_arcsec'``, ``'differential_rotation'``.

    Notes
    -----
    HMI magnetograms do not require PSF deconvolution or degradation
    correction unlike AIA EUV images.
    """

    def __init__(self, config: dict) -> None:
        self.config = config

    def process_file(
        self,
        file: str,
        files_to_preprocess: pd.Series,
        path_input: Path,
        path_output: Path,
    ) -> None:
        """
        Preprocess a single HMI FITS file; designed for parallel execution.

        Validates the file, then runs the full preprocessing pipeline for each
        target date mapped to this file. Files failing validation are skipped
        with a printed warning.

        Parameters
        ----------
        file : str
            Filename of the FITS file to process.
        files_to_preprocess : pd.Series
            Mapping from file names to target dates.
        path_input : Path
            Directory containing input FITS files.
        path_output : Path
            Directory for output files.
        """
        if not file.endswith('.fits'):
            return

        fits_file = path_input / file
        date, product, channel = read_file_name(file, preprocessed=False)

        # One file may serve multiple target dates when used for gap filling
        target_dates = files_to_preprocess.loc[[file]].to_list()

        try:
            hmi_map = sunpy.map.Map(fits_file)
        except Exception:
            print(f'FITS file could not be read: {product} magnetogram {date}')
            return

        if not contains_full_disk(hmi_map):
            print(f'Map does not contain full disk: {product} magnetogram {date}')
            return

        if hmi_map.meta['QUALITY'] != 0:
            print(f'Bad quality: {product} magnetogram {date}')
            return

        # Downsample full-resolution (4096 px) images before processing
        if hmi_map.data.shape[0] == 4096:
            hmi_map = hmi_map.resample([1024, 1024] * u.pixel)

        print(f'Processing {product} magnetogram {date}')

        for target_date in target_dates:
            try:
                preprocessed_image, meta_info = self.preprocess(
                    hmi_map, date, target_date
                )
                save_preprocessed_output(
                    path_output, 'hmi', target_date, preprocessed_image, meta_info
                )
            except Exception:
                print(f'Failed to compute {product} magnetogram {target_date}')
                return


    def preprocess(
        self,
        hmi_map: sunpy.map.Map,
        map_date: datetime,
        target_date: datetime,
    ) -> Tuple[np.ndarray, dict]:
        """
        Execute the full HMI preprocessing pipeline.

        Parameters
        ----------
        hmi_map : sunpy.map.Map
            Input HMI magnetogram map.
        map_date : datetime
            Actual observation time of the input map.
        target_date : datetime
            Target time for the output, used for differential rotation.

        Returns
        -------
        Tuple[np.ndarray, dict]
            Preprocessed magnetogram array (solar north up) and metadata dict
            from the final processed map.
        """
        # Upsample to full resolution before differential rotation
        hmi_map = hmi_map.resample([4096, 4096] * u.pixel)
        hmi_map = self._apply_differential_rotation(hmi_map, map_date, target_date)

        # Downsample to output resolution
        hmi_map = hmi_map.resample([1024, 1024] * u.pixel)

        # Register with zero fill for off-disk (missing) regions
        hmi_map = register_image(hmi_map, missing=0.0)

        # Replace NaN values (e.g., off-disk pixels after rotation) with zero
        hmi_map = sunpy.map.Map(np.nan_to_num(np.array(hmi_map.data)), hmi_map.meta)

        hmi_map = scale_solar_disk_radius(
            hmi_map,
            rsun_target=self.config['target_rsun_arcsec'],
            missing=0.0,
        )

        # SDO images are stored solar-south-up; flip to standard north-up
        img_final = np.flipud(hmi_map.data)

        return img_final, hmi_map.meta

    def _apply_differential_rotation(
        self,
        hmi_map: sunpy.map.Map,
        map_date: datetime,
        target_date: datetime,
    ) -> sunpy.map.Map:
        """
        Rotate the map to ``target_date`` if the time gap exceeds 6 minutes.

        Parameters
        ----------
        hmi_map : sunpy.map.Map
            Input HMI map.
        map_date : datetime
            Actual observation time.
        target_date : datetime
            Target time to rotate to.

        Returns
        -------
        sunpy.map.Map
            Differentially rotated map, or the original if within tolerance.

        Raises
        ------
        ValueError
            If the time gap exceeds 6 minutes and differential rotation is
            disabled in the configuration.
        """
        time_diff = abs(map_date - target_date)
        if time_diff > timedelta(minutes=6):
            if self.config['differential_rotation']:
                return compute_differential_rotation(hmi_map, target_date)
            else:
                raise ValueError(
                    'Time difference between downloaded image and target time '
                    'is too large. Differential rotation must be applied, but '
                    'is disabled. Preprocessing failed.'
                )
        return hmi_map

