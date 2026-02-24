from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple

import astropy.units as u
import numpy as np
import pandas as pd
import sunpy.map
from sunpy.map import contains_full_disk

from aiapy.calibrate import update_pointing, correct_degradation

from solar_image_processing.psf_deconvolution.deconvolve_image import deconvolve_bid
from solar_image_processing.utils.helper_functions import (
    read_file_name,
    save_preprocessed_output,
)
from solar_image_processing.preprocessing.preprocessing_functions import (
    register_image,
    scale_solar_disk_radius,
    compute_differential_rotation,
)


class AIAPreprocessor:
    """
    Preprocessor for AIA (Atmospheric Imaging Assembly) EUV images.

    Implements the full AIA preprocessing pipeline:

    1. Upsample to 4096 px and update pointing.
    2. Apply differential rotation correction if needed.
    3. Downsample to 1024 px and deconvolve with the PSF.
    4. Register (align) the image to solar disk centre.
    5. Normalise solar disk radius.
    6. Correct for instrument degradation.
    7. Normalise by exposure time.
    8. Flip to solar north-up orientation.

    Parameters
    ----------
    pointing_table : pd.DataFrame
        AIA pointing information table from JSOC.
    point_spread_function : np.ndarray
        Rebinned PSF for the specific wavelength.
    correction_table : pd.DataFrame
        Degradation correction table from aiapy.
    config : dict
        Preprocessing configuration. Expected keys:
        ``'target_rsun_arcsec'``, ``'differential_rotation'``, ``'use_gpu'``.
    """

    def __init__(
        self,
        pointing_table: pd.DataFrame,
        point_spread_function: np.ndarray,
        correction_table: pd.DataFrame,
        config: dict,
    ) -> None:
        self.pointing_table = pointing_table
        self.psf = point_spread_function
        self.correction_table = correction_table
        self.config = config

    def process_file(
        self,
        file: str,
        files_to_preprocess: pd.Series,
        path_input: Path,
        path_output: Path,
    ) -> None:
        """
        Preprocess a single AIA FITS file; designed for parallel execution.

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
            aia_map = sunpy.map.Map(fits_file)
        except Exception:
            print(f'FITS file could not be read: {product} {channel} {date}')
            return

        if not contains_full_disk(aia_map):
            print(f'Map does not contain full disk: {product} {channel} {date}')
            return

        if aia_map.meta['QUALITY'] != 0:
            print(f'Bad quality: {product} {channel} {date}')
            return

        # Downsample full-resolution (4096 px) images before processing
        if aia_map.data.shape[0] == 4096:
            aia_map = aia_map.resample([1024, 1024] * u.pixel)

        print(f'Processing {product} {channel} {date}')

        for target_date in target_dates:
            try:
                preprocessed_image, meta_info = self.preprocess(
                    aia_map, date, target_date
                )
                save_preprocessed_output(
                    path_output, channel, target_date, preprocessed_image, meta_info
                )
            except Exception:
                print(f'Failed to compute {product} {channel} {target_date}')
                return

    def preprocess(
        self,
        aia_map: sunpy.map.Map,
        map_date: datetime,
        target_date: datetime,
    ) -> Tuple[np.ndarray, dict]:
        """
        Execute the full AIA preprocessing pipeline.

        Parameters
        ----------
        aia_map : sunpy.map.Map
            Input AIA map (any resolution).
        map_date : datetime
            Actual observation time of the input map.
        target_date : datetime
            Target time for the output, used for differential rotation.

        Returns
        -------
        Tuple[np.ndarray, dict]
            Preprocessed image array (solar north up) and metadata dict
            from the final processed map.
        """
        # Upsample to full resolution before updating pointing
        aia_map = aia_map.resample([4096, 4096] * u.pixel)
        aia_map = update_pointing(aia_map, pointing_table=self.pointing_table)

        aia_map = self._apply_differential_rotation(aia_map, map_date, target_date)

        # Downsample before deconvolution to reduce computation time
        aia_map = aia_map.resample([1024, 1024] * u.pixel)

        aia_map = self._deconvolve(aia_map)
        aia_map = register_image(aia_map)
        aia_map = scale_solar_disk_radius(
            aia_map,
            rsun_target=self.config['target_rsun_arcsec'],
        )
        aia_map = correct_degradation(
            aia_map,
            correction_table=self.correction_table,
        )

        # Normalise by exposure time to obtain DN/s
        img_normalized = aia_map.data / aia_map.exposure_time

        # SDO images are stored solar-south-up; flip to standard north-up
        img_final = np.flipud(img_normalized.value)

        return img_final, aia_map.meta

    def _apply_differential_rotation(
        self,
        aia_map: sunpy.map.Map,
        map_date: datetime,
        target_date: datetime,
    ) -> sunpy.map.Map:
        """
        Rotate the map to ``target_date`` if the time gap exceeds 6 minutes.

        Parameters
        ----------
        aia_map : sunpy.map.Map
            Input AIA map.
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
                return compute_differential_rotation(aia_map, target_date)
            else:
                print(
                    'Time difference between downloaded image and target time '
                    'is too large. Differential rotation must be applied, but '
                    'is disabled. Preprocessing failed.'
                )
                raise ValueError(
                    'Time difference between downloaded image and target time '
                    'is too large. Differential rotation must be applied, but '
                    'is disabled. Preprocessing failed.'
                )
        return aia_map

    def _deconvolve(self, aia_map: sunpy.map.Map) -> sunpy.map.Map:
        """
        Deconvolve image data with the instrument PSF using the BID algorithm.

        Parameters
        ----------
        aia_map : sunpy.map.Map
            Input AIA map at working resolution.

        Returns
        -------
        sunpy.map.Map
            Map with deconvolved data; negative corner artefacts clipped to zero.
        """
        deconvolved = deconvolve_bid(
            aia_map.data,
            self.psf,
            use_gpu=self.config['use_gpu'],
        )
        # Negative values in corners arise from interpolation during deconvolution
        deconvolved[deconvolved < 0] = 0.0
        return sunpy.map.Map(deconvolved, aia_map.meta)


