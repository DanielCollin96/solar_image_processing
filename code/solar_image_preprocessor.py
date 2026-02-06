"""
Solar Image Preprocessor Module.

This module provides functionality to preprocess SDO (Solar Dynamics Observatory)
solar images for scientific analysis. It supports preprocessing at hourly cadence
for both AIA (Atmospheric Imaging Assembly) and HMI (Helioseismic and Magnetic
Imager) instruments.

Key Features:
- Image registration (rotation, scaling, translation alignment)
- PSF deconvolution for AIA images
- Differential rotation correction for time-gap substitution
- Degradation correction for AIA images
- Solar disk radius normalization
- Image cropping and downsampling

Example Usage:
    config = create_aia_preprocess_config(wavelength=171)
    preprocessor = SolarImagePreprocessor(config, paths)
    preprocessor.run(start_date, end_date)
"""

import os
import pickle
from copy import copy
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Union, Tuple, List

import astropy.units as u
import numpy as np
import pandas as pd
import sunpy.map
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.wcs import WCS
from dateutil.relativedelta import relativedelta
from joblib import Parallel, delayed, cpu_count
from skimage.measure import block_reduce
from skimage.transform import resize
from sunpy.coordinates import Helioprojective, propagate_with_solar_surface
from sunpy.map import contains_full_disk

from aiapy.calibrate import update_pointing, correct_degradation

from deconvolve_image import deconvolve_bid
from rebin_psf import rebin_psf
from utils import (
    read_file_name,
    check_file_quality,
    find_missing_preprocessed_dates,
    load_existing_raw_files,
    load_config_data,
    find_files_to_preprocess,
    check_completeness_of_preprocessed_images,
    find_missing_cropped_dates,
    load_existing_preprocessed_dates,
)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class PreprocessConfiguration:
    """
    Configuration settings for solar image preprocessing.

    Parameters
    ----------
    channel : str
        Channel identifier (e.g., 'aia_171', 'aia_193', 'aia_211', 'hmi').
    input_resolution : int
        Expected resolution of input FITS images (default: 1024).
    output_resolution : int
        Target resolution after preprocessing (default: 1024).
    target_rsun_arcsec : float
        Target solar radius in arcseconds for normalization (default: 976.0).
    use_gpu : bool
        Whether to use GPU for deconvolution (default: False).
    """

    channel: str
    input_resolution: int = 1024
    output_resolution: int = 1024
    target_rsun_arcsec: float = 976.0
    use_gpu: bool = False

    @property
    def is_aia(self) -> bool:
        """Check if configuration is for AIA data."""
        return self.channel.startswith('aia_')

    @property
    def is_hmi(self) -> bool:
        """Check if configuration is for HMI data."""
        return self.channel == 'hmi'

    @property
    def wavelength(self) -> Optional[int]:
        """Extract wavelength from AIA channel string."""
        if self.is_aia:
            return int(self.channel.split('_')[1])
        return None

    @property
    def channel_str(self) -> str:
        """
        Get channel string for file naming.

        Returns wavelength for AIA (e.g., '171') or 'hmi' for HMI.
        """
        if self.is_aia:
            return self.channel.split('_')[1]
        return 'hmi'



# =============================================================================
# Image Processing Functions
# =============================================================================


def register_image(
    smap: sunpy.map.Map,
    missing: Optional[float] = None,
    arcsec_pix_target: Optional[float] = None
) -> sunpy.map.Map:
    """
    Register an SDO image by rotating, scaling, and centering.

    This function aligns solar images to a common reference frame by:
    1. Rotating to align solar north with image vertical
    2. Scaling to achieve target pixel scale
    3. Centering the solar disk in the image

    Parameters
    ----------
    smap : sunpy.map.Map
        Input SunPy map to register.
    missing : float, optional
        Value to use for missing/interpolated pixels.
        Defaults to minimum value in the map.
    arcsec_pix_target : float, optional
        Target pixel scale in arcseconds per pixel.
        Defaults to 0.6 arcsec/pix scaled by resolution ratio.

    Returns
    -------
    sunpy.map.Map
        Registered map with updated metadata.

    Notes
    -----
    The level number is updated to 1.5 to indicate registration.
    For images where shape is not a power of 2, explicit arcsec_pix_target
    should be provided for accurate scaling.
    """
    orig_shape = smap.data.shape[0]

    # Calculate default pixel scale based on resolution ratio to 4096
    if arcsec_pix_target is None:
        downsample_factor = 4096 / orig_shape
        arcsec_pix_target = 0.6 * downsample_factor

    if arcsec_pix_target is None and np.log2(smap.data.shape[0]) % 1 > 0:
        print('Warning: Map shape is not a power of 2. '
              'Please specify a target arcsec per pixel size.')

    scale = arcsec_pix_target * u.arcsec
    scale_factor = smap.scale[0] / scale
    missing = smap.min() if missing is None else missing

    # Rotate and scale the map using scipy interpolation
    tempmap = smap.rotate(
        recenter=True,
        scale=scale_factor.value,
        order=3,
        missing=missing,
        method='scipy',
    )

    # Extract center region from padded output
    # crpix1 and crpix2 are equal due to recenter=True
    center = np.floor(tempmap.meta["crpix1"])
    range_side = (center + np.array([-1, 1]) * smap.data.shape[0] / 2) * u.pix
    newmap = tempmap.submap(
        u.Quantity([range_side[0], range_side[0]]),
        top_right=u.Quantity([range_side[1], range_side[1]]) - 1 * u.pix,
    )

    # Update metadata
    newmap.meta["r_sun"] = newmap.meta["rsun_obs"] / newmap.meta["cdelt1"]
    newmap.meta["lvl_num"] = 1.5
    newmap.meta["bitpix"] = -64

    # Handle size mismatch: crop if larger, pad if smaller
    newmap = _adjust_map_size(newmap, orig_shape)

    # Update reference pixel to image center
    newmap.meta["crpix1"] = orig_shape / 2 + 0.5
    newmap.meta["crpix2"] = orig_shape / 2 + 0.5

    return newmap


def _adjust_map_size(smap: sunpy.map.Map, target_shape: int) -> sunpy.map.Map:
    """
    Adjust map size to match target shape by cropping or padding.

    Parameters
    ----------
    smap : sunpy.map.Map
        Input map to adjust.
    target_shape : int
        Target size for square output.

    Returns
    -------
    sunpy.map.Map
        Adjusted map with correct dimensions.
    """
    current_shape = smap.data.shape[0]

    if current_shape > target_shape:
        # Crop from center
        center_pixel = current_shape / 2 - 0.5
        half_size = target_shape / 2
        cutout_start = int(center_pixel - half_size + 0.5)
        cutout_end = int(center_pixel + half_size + 0.5)
        new_data = smap.data[cutout_start:cutout_end, cutout_start:cutout_end]
        return sunpy.map.Map(new_data, smap.meta)

    elif current_shape < target_shape:
        # Pad symmetrically with zeros
        pad_width = int((target_shape - current_shape) / 2)
        new_data = np.pad(
            smap.data,
            pad_width,
            mode='constant',
            constant_values=0.
        )
        return sunpy.map.Map(new_data, smap.meta)

    return smap


def scale_solar_disk_radius(
    smap: sunpy.map.Map,
    rsun_target: float = 976.0,
    missing: Optional[float] = None
) -> sunpy.map.Map:
    """
    Scale a registered map to achieve a fixed solar disk radius.

    This normalization ensures all images have the same apparent solar disk
    size, which is essential for consistent analysis across different
    observation times (solar distance varies throughout the year).

    Parameters
    ----------
    smap : sunpy.map.Map
        Input registered map to scale.
    rsun_target : float, optional
        Target solar radius in arcseconds (default: 976.0).
        This value determines the pixel radius of the solar disk.
    missing : float, optional
        Value for interpolated pixels. Defaults to data minimum.

    Returns
    -------
    sunpy.map.Map
        Scaled map with normalized solar disk radius.
    """
    orig_shape = smap.data.shape[0]
    scale_factor = rsun_target / smap.meta['RSUN_OBS']
    missing = smap.data.min() if missing is None else missing

    temp_map = smap.rotate(
        scale=scale_factor,
        order=3,
        missing=missing,
        method='scipy'
    )

    # Adjust output size to match original
    new_img_data = _extract_or_pad_data(temp_map.data, orig_shape)
    return sunpy.map.Map(new_img_data, temp_map.meta)


def _extract_or_pad_data(data: np.ndarray, target_shape: int) -> np.ndarray:
    """
    Extract center region or pad data to match target shape.

    Parameters
    ----------
    data : np.ndarray
        Input 2D array.
    target_shape : int
        Target size for square output.

    Returns
    -------
    np.ndarray
        Adjusted array with target dimensions.
    """
    current_shape = data.shape[0]

    if current_shape > target_shape:
        center_pixel = current_shape / 2 - 0.5
        half_size = target_shape / 2
        cutout_start = int(center_pixel - half_size + 0.5)
        cutout_end = int(center_pixel + half_size + 0.5)
        return data[cutout_start:cutout_end, cutout_start:cutout_end]

    elif current_shape < target_shape:
        pad_width = int((target_shape - current_shape) / 2)
        return np.pad(data, pad_width, mode='constant', constant_values=0.)

    return data


def compute_differential_rotation(
    smap: sunpy.map.Map,
    target_date: datetime
) -> sunpy.map.Map:
    """
    Apply differential rotation to match a target observation time.

    Solar features rotate at different rates depending on latitude
    (differential rotation). This function reprojects an image to
    account for this rotation, allowing images from nearby times
    to substitute for missing observations.

    Parameters
    ----------
    smap : sunpy.map.Map
        Input map to rotate.
    target_date : datetime
        Target observation time to rotate to.

    Returns
    -------
    sunpy.map.Map
        Reprojected map adjusted for differential rotation.

    Notes
    -----
    Uses SunPy's propagate_with_solar_surface context manager
    for accurate differential rotation modeling.
    """
    # Create output coordinate frame at target time
    out_frame = Helioprojective(
        observer=smap.observer_coordinate,
        obstime=Time(target_date),
        rsun=smap.coordinate_frame.rsun
    )
    out_center = SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=out_frame)

    # Build WCS header for output
    header = sunpy.map.make_fitswcs_header(
        smap.data.shape,
        out_center,
        reference_pixel=u.Quantity(smap.reference_pixel),
        scale=u.Quantity(smap.scale),
        rotation_matrix=smap.rotation_matrix,
        instrument=smap.instrument,
        exposure=smap.exposure_time
    )
    out_wcs = WCS(header)

    # Reproject with differential rotation
    with propagate_with_solar_surface():
        smap_reprojected = smap.reproject_to(out_wcs)

    # Preserve original metadata
    return sunpy.map.Map(smap_reprojected.data, smap.meta)


# =============================================================================
# Instrument-Specific Preprocessors
# =============================================================================


class AIAPreprocessor:
    """
    Preprocessor for AIA (Atmospheric Imaging Assembly) EUV images.

    This class implements the full AIA preprocessing pipeline including:
    - Pointing update
    - Differential rotation correction
    - PSF deconvolution
    - Image registration
    - Solar disk radius normalization
    - Degradation correction
    - Exposure time normalization

    Parameters
    ----------
    pointing_table : pandas.DataFrame
        AIA pointing information table from JSOC.
    point_spread_function : np.ndarray
        Rebinned PSF for the specific wavelength.
    correction_table : pandas.DataFrame
        Degradation correction table from aiapy.
    config : PreprocessConfiguration
        Preprocessing configuration settings.

    Example
    -------
    >>> preprocessor = AIAPreprocessor(pointing, psf, correction, config)
    >>> image, metadata = preprocessor.preprocess(aia_map, map_date, target_date)
    """

    # Resolution for full-size processing steps
    FULL_RESOLUTION = 4096
    # Resolution for deconvolution (speed/accuracy tradeoff)
    DECONV_RESOLUTION = 1024
    # Time threshold for applying differential rotation (minutes)
    DIFF_ROT_THRESHOLD_MINUTES = 6

    def __init__(
        self,
        pointing_table: pd.DataFrame,
        point_spread_function: np.ndarray,
        correction_table: pd.DataFrame,
        config: PreprocessConfiguration
    ) -> None:
        self.pointing_table = pointing_table
        self.psf = point_spread_function
        self.correction_table = correction_table
        self.config = config

    def preprocess(
        self,
        aia_map: sunpy.map.Map,
        map_date: datetime,
        target_date: datetime
    ) -> Tuple[np.ndarray, dict]:
        """
        Execute the full AIA preprocessing pipeline.

        Parameters
        ----------
        aia_map : sunpy.map.Map
            Input AIA map (can be any resolution).
        map_date : datetime
            Actual observation time of the input map.
        target_date : datetime
            Target time for the output (for differential rotation).

        Returns
        -------
        Tuple[np.ndarray, dict]
            Preprocessed image array (flipped to solar north up) and
            metadata dictionary from the final processed map.
        """
        # Step 1: Upsample to full resolution for pointing update
        aia_map = self._upsample_for_pointing(aia_map)

        # Step 2: Update pointing information
        aia_map = update_pointing(aia_map, pointing_table=self.pointing_table)

        # Step 3: Apply differential rotation if needed
        aia_map = self._apply_differential_rotation(aia_map, map_date, target_date)

        # Step 4: Downsample for faster deconvolution
        aia_map = self._downsample_for_deconvolution(aia_map)

        # Step 5: Deconvolve with PSF
        aia_map = self._deconvolve(aia_map)

        # Step 6: Register (align) the image
        aia_map = register_image(aia_map)

        # Step 7: Normalize solar disk radius
        aia_map = scale_solar_disk_radius(
            aia_map,
            rsun_target=self.config.target_rsun_arcsec
        )

        # Step 8: Correct for instrument degradation
        aia_map = correct_degradation(
            aia_map,
            correction_table=self.correction_table
        )

        # Step 9: Normalize by exposure time
        img_normalized = aia_map.data / aia_map.exposure_time

        # Step 10: Flip to solar north up orientation
        # SDO data is provided with south up, flip for standard orientation
        img_final = np.flipud(img_normalized.value)

        return img_final, aia_map.meta

    def _upsample_for_pointing(self, aia_map: sunpy.map.Map) -> sunpy.map.Map:
        """Upsample to 4096x4096 for accurate pointing update."""
        new_dimensions = [self.FULL_RESOLUTION, self.FULL_RESOLUTION] * u.pixel
        return aia_map.resample(new_dimensions)

    def _apply_differential_rotation(
        self,
        aia_map: sunpy.map.Map,
        map_date: datetime,
        target_date: datetime
    ) -> sunpy.map.Map:
        """Apply differential rotation if time gap exceeds threshold."""
        time_diff = abs(map_date - target_date)
        if time_diff > timedelta(minutes=self.DIFF_ROT_THRESHOLD_MINUTES):
            return compute_differential_rotation(aia_map, target_date)
        return aia_map

    def _downsample_for_deconvolution(
        self,
        aia_map: sunpy.map.Map
    ) -> sunpy.map.Map:
        """Downsample to 1024x1024 for faster deconvolution."""
        new_dimensions = [self.DECONV_RESOLUTION, self.DECONV_RESOLUTION] * u.pixel
        return aia_map.resample(new_dimensions)

    def _deconvolve(self, aia_map: sunpy.map.Map) -> sunpy.map.Map:
        """
        Deconvolve image with PSF using BID algorithm.

        Negative values in corners (from interpolation) are set to zero.
        """
        deconvolved = deconvolve_bid(
            aia_map.data,
            self.psf,
            use_gpu=self.config.use_gpu
        )
        # Remove negative values introduced by deconvolution in image corners
        deconvolved[deconvolved < 0] = 0.0
        return sunpy.map.Map(deconvolved, aia_map.meta)


class HMIPreprocessor:
    """
    Preprocessor for HMI (Helioseismic and Magnetic Imager) magnetograms.

    This class implements the HMI preprocessing pipeline including:
    - Differential rotation correction
    - Image registration
    - Solar disk radius normalization
    - NaN handling

    Parameters
    ----------
    config : PreprocessConfiguration
        Preprocessing configuration settings.

    Notes
    -----
    HMI magnetograms do not require PSF deconvolution or degradation
    correction like AIA EUV images.

    Example
    -------
    >>> preprocessor = HMIPreprocessor(config)
    >>> image, metadata = preprocessor.preprocess(hmi_map, map_date, target_date)
    """

    FULL_RESOLUTION = 4096
    OUTPUT_RESOLUTION = 1024
    DIFF_ROT_THRESHOLD_MINUTES = 6

    def __init__(self, config: PreprocessConfiguration) -> None:
        self.config = config

    def preprocess(
        self,
        hmi_map: sunpy.map.Map,
        map_date: datetime,
        target_date: datetime
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
            Target time for the output (for differential rotation).

        Returns
        -------
        Tuple[np.ndarray, dict]
            Preprocessed magnetogram array (flipped to solar north up) and
            metadata dictionary from the final processed map.
        """
        # Step 1: Upsample to full resolution
        new_dimensions = [self.FULL_RESOLUTION, self.FULL_RESOLUTION] * u.pixel
        hmi_map = hmi_map.resample(new_dimensions)

        # Step 2: Apply differential rotation if needed
        hmi_map = self._apply_differential_rotation(hmi_map, map_date, target_date)

        # Step 3: Downsample to output resolution
        new_dimensions = [self.OUTPUT_RESOLUTION, self.OUTPUT_RESOLUTION] * u.pixel
        hmi_map = hmi_map.resample(new_dimensions)

        # Step 4: Register with zero fill for off-disk regions
        hmi_map = register_image(hmi_map, missing=0.0)

        # Step 5: Handle NaN values (replace with zero)
        hmi_map = self._remove_nan_values(hmi_map)

        # Step 6: Normalize solar disk radius
        hmi_map = scale_solar_disk_radius(
            hmi_map,
            rsun_target=self.config.target_rsun_arcsec,
            missing=0.0
        )

        # Step 7: Flip to solar north up orientation
        img_final = np.flipud(hmi_map.data)

        return img_final, hmi_map.meta

    def _apply_differential_rotation(
        self,
        hmi_map: sunpy.map.Map,
        map_date: datetime,
        target_date: datetime
    ) -> sunpy.map.Map:
        """Apply differential rotation if time gap exceeds threshold."""
        time_diff = abs(map_date - target_date)
        if time_diff > timedelta(minutes=self.DIFF_ROT_THRESHOLD_MINUTES):
            return compute_differential_rotation(hmi_map, target_date)
        return hmi_map

    def _remove_nan_values(self, hmi_map: sunpy.map.Map) -> sunpy.map.Map:
        """Replace NaN values with zero."""
        data = np.nan_to_num(np.array(hmi_map.data))
        return sunpy.map.Map(data, hmi_map.meta)


# =============================================================================
# Parallel Processing Workers
# =============================================================================


def _process_single_aia_file(
    file: str,
    files_to_preprocess: pd.Series,
    path_input: Path,
    path_output: Path,
    pointing_table: pd.DataFrame,
    psf: np.ndarray,
    correction_table: pd.DataFrame,
    config: PreprocessConfiguration
) -> None:
    """
    Process a single AIA FITS file (worker function for parallel execution).

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
    pointing_table : pd.DataFrame
        AIA pointing table.
    psf : np.ndarray
        Rebinned PSF for this wavelength.
    correction_table : pd.DataFrame
        Degradation correction table.
    config : PreprocessConfiguration
        Preprocessing configuration.
    """
    if not file.endswith('.fits'):
        return

    fits_file = path_input / file
    date, product, channel = read_file_name(file, preprocessed=False)

    # Get all target dates for this file (may be multiple due to gap filling)
    target_dates = files_to_preprocess.loc[[file]].to_list()

    # Load and validate the map
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

    # Downsample if at full resolution
    if aia_map.data.shape[0] == 4096:
        aia_map = aia_map.resample([1024, 1024] * u.pixel)

    print(f'Processing {product} {channel} {date}')

    # Create preprocessor and process for each target date
    preprocessor = AIAPreprocessor(
        pointing_table, psf, correction_table, config
    )

    for target_date in target_dates:
        preprocessed_image, meta_info = preprocessor.preprocess(
            aia_map, date, target_date
        )
        _save_preprocessed_output(
            path_output, channel, target_date, preprocessed_image, meta_info
        )


def _process_single_hmi_file(
    file: str,
    files_to_preprocess: pd.Series,
    path_input: Path,
    path_output: Path,
    config: PreprocessConfiguration
) -> None:
    """
    Process a single HMI FITS file (worker function for parallel execution).

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
    config : PreprocessConfiguration
        Preprocessing configuration.
    """
    if not file.endswith('.fits'):
        return

    fits_file = path_input / file
    date, product, channel = read_file_name(file, preprocessed=False)

    # Get all target dates for this file
    target_dates = files_to_preprocess.loc[[file]].to_list()

    # Load and validate the map
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

    # Downsample if at full resolution
    if hmi_map.data.shape[0] == 4096:
        hmi_map = hmi_map.resample([1024, 1024] * u.pixel)

    print(f'Processing {product} magnetogram {date}')

    # Create preprocessor and process for each target date
    preprocessor = HMIPreprocessor(config)

    for target_date in target_dates:
        preprocessed_image, meta_info = preprocessor.preprocess(
            hmi_map, date, target_date
        )
        _save_preprocessed_output(
            path_output, 'hmi', target_date, preprocessed_image, meta_info
        )


def _save_preprocessed_output(
    path_output: Path,
    channel: str,
    target_date: datetime,
    image: np.ndarray,
    metadata: dict
) -> None:
    """
    Save preprocessed image and metadata to disk.

    Parameters
    ----------
    path_output : Path
        Output directory.
    channel : str
        Channel identifier for filename.
    target_date : datetime
        Target date for filename.
    image : np.ndarray
        Preprocessed image data.
    metadata : dict
        Image metadata.
    """
    date_str = target_date.strftime('%Y-%m-%d_%H:%M')
    base_name = f'{channel}_{date_str}'

    np.save(path_output / f'{base_name}.npy', image)
    with open(path_output / f'{base_name}_meta.pickle', 'wb') as f:
        pickle.dump(metadata, f)

    print(f'Saved {base_name}.npy')


def _crop_single_image(
    file: str,
    path_input: Path,
    path_output: Path,
    crop_config: dict
) -> None:
    """
    Crop and downsample a single preprocessed image.

    Parameters
    ----------
    file : str
        Filename of the .npy file to process.
    path_input : Path
        Directory containing preprocessed images.
    path_output : Path
        Directory for cropped output.
    crop_config : dict
        Cropping configuration.
    """
    if not file.endswith('.npy'):
        return

    print(f'Processing image {file}')

    img = np.load(path_input / file)
    current_resolution = img.shape[0]

    # Validate resolution compatibility
    if current_resolution % crop_config['downsample_resolution'] != 0:
        raise ValueError(
            f'Image resolution {current_resolution} must be divisible by '
            f'target resolution {crop_config['downsample_resolution']}.'
        )

    # Downsample by summing in blocks (preserves total flux)
    downsample_factor = current_resolution // crop_config['downsample_resolution']
    img = block_reduce(img, (downsample_factor, downsample_factor), np.sum)

    # Crop image based on configuration
    img = _apply_crop(img, file, path_input, crop_config, downsample_factor)

    # Resize to final dimensions if specified
    if crop_config['resize_cropped'] is not None:
        img = resize(
            img,
            (crop_config['resize_cropped'], crop_config['resize_cropped']),
            order=3,
            mode='constant',
            cval=0
        )

    # Save as float32 for efficiency
    np.save(path_output / file, img.astype('float32'))


def _apply_crop(
    img: np.ndarray,
    file: str,
    path_input: Path,
    crop_config: dict,
    downsample_factor: int
) -> np.ndarray:
    """
    Apply cropping based on configuration mode.

    Parameters
    ----------
    img : np.ndarray
        Downsampled image.
    file : str
        Original filename (for metadata lookup).
    path_input : Path
        Input directory (for metadata file).
    crop_config : dict
        Cropping configuration.
    downsample_factor : int
        Factor by which image was downsampled.

    Returns
    -------
    np.ndarray
        Cropped image.
    """
    if isinstance(crop_config['crop_mode'], int):
        # Fixed square crop
        cut_pixels = int((img.shape[0] - crop_config['crop_mode']) / 2)
        return img[cut_pixels:-cut_pixels, cut_pixels:-cut_pixels]

    elif crop_config['crop_mode'] == 'disk':
        # Crop to solar disk boundary
        meta_file = path_input / file.replace('.npy', '_meta.pickle')
        with open(meta_file, 'rb') as f:
            meta_data = pickle.load(f)

        # Calculate disk radius in downsampled pixels
        downsampled_scale = meta_data['cdelt1'] * downsample_factor
        sun_radius_pixels = meta_data['rsun_obs'] / downsampled_scale
        cut_pixels = int(np.round((img.shape[0] - sun_radius_pixels * 2) / 2, 2))
        return img[cut_pixels:-cut_pixels, cut_pixels:-cut_pixels]

    return img


# =============================================================================
# Main Orchestrator Classes
# =============================================================================


class SolarImagePreprocessor:
    """
    Main orchestrator for preprocessing SDO solar images.

    This class manages the full preprocessing workflow including:
    - Directory structure creation
    - Finding missing preprocessed dates
    - Parallel preprocessing of images
    - Progress tracking and validation

    Parameters
    ----------
    channels : List[str]
        List of channels to process (e.g., ['aia_171', 'aia_193', 'hmi']).
    paths : dict
        Dictionary with keys 'raw', 'preprocessed', 'config' pointing to
        respective directories as Path objects.
    use_gpu : bool, optional
        Whether to use GPU for AIA deconvolution (default: False).

    Example
    -------
    >>> paths = {
    ...     'raw': Path('/data/raw/SDO'),
    ...     'preprocessed': Path('/data/preprocessed'),
    ...     'config': Path('/data/config')
    ... }
    >>> preprocessor = SolarImagePreprocessor(['aia_171', 'hmi'], paths)
    >>> preprocessor.run(datetime(2020, 1, 1), datetime(2020, 1, 31))
    """

    def __init__(
        self,
        channels: List[str],
        paths: dict,
        use_gpu: bool = False
    ) -> None:
        self.channels = channels
        self.paths = paths
        self.use_gpu = use_gpu

        # Create configuration for each channel
        self.configs = {
            channel: PreprocessConfiguration(channel=channel, use_gpu=use_gpu)
            for channel in channels
        }

    def run(
        self,
        start: datetime,
        end: datetime,
        load_preprocessing_fails: bool = False,
        overwrite_existing: bool = False
    ) -> None:
        """
        Run preprocessing for all configured channels.

        Parameters
        ----------
        start : datetime
            Start date for preprocessing.
        end : datetime
            End date for preprocessing.
        load_preprocessing_fails : bool, optional
            Whether to load and skip previously failed files (default: False).
        overwrite_existing : bool, optional
            Whether to reprocess existing files (default: False).
        """
        # Ensure output directories exist
        self._create_output_directories(start, end)

        for channel in self.channels:
            print(f'\n{"="*60}')
            print(f'Processing channel: {channel}')
            print(f'{"="*60}')

            self._process_channel(
                channel, start, end,
                load_preprocessing_fails, overwrite_existing
            )

    def _create_output_directories(
        self,
        start: datetime,
        end: datetime
    ) -> None:
        """Create directory structure for preprocessed output."""
        preprocessed_path = Path(self.paths['preprocessed'])
        preprocessed_path.mkdir(parents=True, exist_ok=True)

        for channel in self.channels:
            channel_path = preprocessed_path / channel
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
        load_preprocessing_fails: bool,
        overwrite_existing: bool
    ) -> None:
        """Process a single channel across all months."""
        config = self.configs[channel]

        # Determine input path based on channel type
        if config.is_aia:
            path_raw = self.paths['raw'] / 'AIA' / str(config.wavelength)
        else:
            path_raw = self.paths['raw'] / 'HMI' / 'magnetogram'

        path_preprocessed = Path(self.paths['preprocessed']) / channel
        current_month = datetime(start.year, start.month, 1)
        dates_to_check = []

        while current_month <= end:
            dates_to_check.extend(
                self._process_month(
                    channel, config, current_month,
                    path_raw, path_preprocessed,
                    load_preprocessing_fails, overwrite_existing
                )
            )
            current_month += relativedelta(months=1)

        # Report final status
        print('\nAll preprocessing complete?')
        print(f'  {len(dates_to_check) == 0}')
        if dates_to_check:
            print(f'  Dates to check: {dates_to_check}')

    def _process_month(
        self,
        channel: str,
        config: PreprocessConfiguration,
        month: datetime,
        path_raw: Path,
        path_preprocessed: Path,
        load_preprocessing_fails: bool,
        overwrite_existing: bool
    ) -> List[datetime]:
        """
        Process all images for a single month.

        Returns list of dates that need investigation.
        """
        print(f'\nProcessing {month.strftime("%Y/%m")}')

        path_raw_month = path_raw / month.strftime('%Y/%m')
        path_output = path_preprocessed / month.strftime('%Y/%m')

        # Find existing raw files
        existing_raw_files = load_existing_raw_files(path_raw_month)

        # Find missing preprocessed dates
        missing_dates, _ = find_missing_preprocessed_dates(
            month, path_output, channel, overwrite_existing
        )

        # Load previously failed files if requested
        files_to_exclude = self._load_exclusion_list(
            path_output, load_preprocessing_fails
        )
        if load_preprocessing_fails and not files_to_exclude.empty:
            missing_dates = missing_dates.difference(files_to_exclude.index)

        # Find files that can be preprocessed
        if len(missing_dates) > 0:
            files_to_preprocess, new_exclusions = find_files_to_preprocess(
                missing_dates, existing_raw_files, path_raw_month
            )

            # Update exclusion list
            if not files_to_exclude.empty:
                new_exclusions = pd.concat([files_to_exclude, new_exclusions])
            new_exclusions.to_csv(path_output / 'preprocessing_fails.csv')
        else:
            files_to_preprocess = pd.Series(dtype=object)
            new_exclusions = files_to_exclude

        # Run preprocessing
        if len(files_to_preprocess) > 0:
            self._run_parallel_preprocessing(
                channel, config, files_to_preprocess,
                path_raw_month, path_output, month
            )

        # Validate completeness
        _, dates_to_check = check_completeness_of_preprocessed_images(
            new_exclusions, month, path_output, channel
        )

        return dates_to_check

    def _load_exclusion_list(
        self,
        path_output: Path,
        load_fails: bool
    ) -> pd.DataFrame:
        """Load list of previously failed preprocessing attempts."""
        if not load_fails:
            return pd.DataFrame(columns=['bad', 'missing_raw'])

        try:
            return pd.read_csv(
                path_output / 'preprocessing_fails.csv',
                index_col=0,
                parse_dates=[0]
            )
        except FileNotFoundError:
            return pd.DataFrame(columns=['bad', 'missing_raw'])

    def _run_parallel_preprocessing(
        self,
        channel: str,
        config: PreprocessConfiguration,
        files_to_preprocess: pd.Series,
        path_input: Path,
        path_output: Path,
        month: datetime
    ) -> None:
        """Execute parallel preprocessing for a batch of files."""
        n_cpus = cpu_count()
        n_jobs = max(1, n_cpus // 2)
        print(f'Starting preprocessing with {n_jobs} workers')

        unique_files = files_to_preprocess.index.unique()

        if config.is_hmi:
            Parallel(n_jobs=n_jobs)(
                delayed(_process_single_hmi_file)(
                    file, files_to_preprocess, path_input, path_output, config
                )
                for file in unique_files
            )
        else:
            # Load AIA-specific calibration data
            psf, correction_table, pointing_table = load_config_data(
                self.paths['config'], str(config.wavelength), month
            )
            Parallel(n_jobs=n_jobs)(
                delayed(_process_single_aia_file)(
                    file, files_to_preprocess, path_input, path_output,
                    pointing_table, psf, correction_table, config
                )
                for file in unique_files
            )


class ImageCropper:
    """
    Orchestrator for cropping preprocessed images.

    This class handles the workflow for cropping preprocessed images to
    smaller regions of interest, with optional downsampling and resizing.

    Parameters
    ----------
    channels : List[str]
        List of channels to process.
    paths : dict
        Dictionary with keys 'preprocessed' and 'cropped' pointing to
        respective directories as Path objects.
    crop_config : dict
        Configuration for cropping operations.

    Example
    -------
    >>> paths = {
    ...     'preprocessed': Path('/data/preprocessed'),
    ...     'cropped': Path('/data/cropped')
    ... }
    >>> crop_config = {
    ...    'downsample_resolution': 256,
    ...    'crop_mode': 'disk',
    ...    'resize_cropped': None
    ...}
    >>> cropper = ImageCropper(['aia_171'], paths, crop_config)
    >>> cropper.run(datetime(2020, 1, 1), datetime(2020, 1, 31))
    """

    def __init__(
        self,
        channels: List[str],
        paths: dict,
        crop_config: dict
    ) -> None:
        self.channels = channels
        self.paths = paths
        self.crop_config = crop_config

    def run(self, start: datetime, end: datetime) -> None:
        """
        Run cropping for all configured channels.

        Parameters
        ----------
        start : datetime
            Start date for processing.
        end : datetime
            End date for processing.
        """
        self._create_output_directories(start, end)

        for channel in self.channels:
            print(f'\n{"="*60}')
            print(f'Cropping channel: {channel}')
            print(f'{"="*60}')

            self._process_channel(channel, start, end)

    def _create_output_directories(
        self,
        start: datetime,
        end: datetime
    ) -> None:
        """Create directory structure for cropped output."""
        cropped_path = Path(self.paths['cropped'])
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
        end: datetime
    ) -> None:
        """Process a single channel across all months."""
        path_preprocessed = Path(self.paths['preprocessed']) / channel
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

        # Report final status
        print('\nAll cropping complete?')
        print(f'  {len(dates_to_check) == 0}')
        if dates_to_check:
            print(f'  Dates to check: {dates_to_check}')

    def _process_month(
        self,
        channel: str,
        month: datetime,
        path_preprocessed: Path,
        path_cropped: Path
    ) -> List[datetime]:
        """Process cropping for a single month."""
        print(f'\nProcessing {month.strftime("%Y/%m")}')

        path_input = path_preprocessed / month.strftime('%Y/%m')
        path_output = path_cropped / month.strftime('%Y/%m')

        # Find existing preprocessed and cropped dates
        existing_preprocessed = load_existing_preprocessed_dates(
            path_input, channel
        )
        missing_cropped, existing_cropped, _ = find_missing_cropped_dates(
            month, path_output, channel
        )

        # Determine which files need cropping
        dates_to_crop = existing_preprocessed.difference(existing_cropped)

        if len(dates_to_crop) > 0:
            files_to_crop = self._build_file_list(channel, dates_to_crop)
            self._run_parallel_cropping(files_to_crop, path_input, path_output)

        # Validate completeness
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

    def _build_file_list(
        self,
        channel: str,
        dates: pd.Index
    ) -> List[str]:
        """Build list of filenames from dates."""
        channel_str = channel.split('_')[1] if 'aia' in channel else channel
        return [
            f'{channel_str}_{date.strftime("%Y-%m-%d_%H:%M")}.npy'
            for date in dates
        ]

    def _run_parallel_cropping(
        self,
        files: List[str],
        path_input: Path,
        path_output: Path
    ) -> None:
        """Execute parallel cropping for a batch of files."""
        n_cpus = cpu_count()
        n_jobs = max(1, n_cpus // 2)
        print(f'Starting cropping with {n_jobs} workers')

        Parallel(n_jobs=n_jobs)(
            delayed(_crop_single_image)(
                file, path_input, path_output, self.crop_config
            )
            for file in files
        )


# =============================================================================
# Convenience Functions
# =============================================================================


def create_aia_preprocess_config(
    wavelength: int,
    use_gpu: bool = False,
    target_rsun: float = 976.0
) -> PreprocessConfiguration:
    """
    Create preprocessing configuration for AIA EUV images.

    Parameters
    ----------
    wavelength : int
        AIA wavelength in Angstroms (e.g., 171, 193, 211).
    use_gpu : bool, optional
        Whether to use GPU for deconvolution (default: False).
    target_rsun : float, optional
        Target solar radius in arcseconds (default: 976.0).

    Returns
    -------
    PreprocessConfiguration
        Configuration object for AIA preprocessing.
    """
    return PreprocessConfiguration(
        channel=f'aia_{wavelength}',
        use_gpu=use_gpu,
        target_rsun_arcsec=target_rsun
    )


def create_hmi_preprocess_config(
    target_rsun: float = 976.0
) -> PreprocessConfiguration:
    """
    Create preprocessing configuration for HMI magnetograms.

    Parameters
    ----------
    target_rsun : float, optional
        Target solar radius in arcseconds (default: 976.0).

    Returns
    -------
    PreprocessConfiguration
        Configuration object for HMI preprocessing.
    """
    return PreprocessConfiguration(
        channel='hmi',
        use_gpu=False,
        target_rsun_arcsec=target_rsun
    )


