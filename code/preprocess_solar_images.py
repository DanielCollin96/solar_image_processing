#!/usr/bin/env python
"""
Solar Image Preprocessing Script.

This script provides the main entry point for preprocessing SDO (Solar Dynamics
Observatory) solar images. It demonstrates typical usage of the preprocessor
and can be customized by modifying the configuration parameters.

The preprocessing pipeline includes:
- Image registration and alignment
- PSF deconvolution (AIA only)
- Degradation correction (AIA only)
- Differential rotation correction
- Solar disk radius normalization
- Optional image cropping and downsampling

Example Usage:
    python preprocess_solar_images.py

See Also:
    src/solar_data/solar_image_preprocessor.py : Module containing preprocessor classes
    download_solar_images.py : Companion script for downloading images
"""

import sys
from datetime import datetime
from pathlib import Path
import os

from solar_image_preprocessor import (
    SolarImagePreprocessor,
    ImageCropper
)


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """
    Main entry point for running the solar image preprocessor.

    This function demonstrates typical usage of the preprocessor and can
    be customized by modifying the configuration parameters below.
    """
    # -------------------------------------------------------------------------
    # Configuration Parameters
    # -------------------------------------------------------------------------

    # Date range for preprocessing
    start_date = datetime(2010, 5, 1)
    end_date = datetime(2024, 6, 30, 23)

    # Channels to process
    # Options: 'aia_171', 'aia_193', 'aia_211', 'hmi'
    channels = ['aia_171', 'aia_193', 'aia_211', 'hmi']

    # Whether to use GPU for AIA deconvolution
    # Requires CUDA-capable GPU and cupy package
    use_gpu = False

    # Cropping configuration
    # - downsample_resolution: Target size after downsampling
    # - crop_mode: Integer for fixed crop size, or 'disk' for solar disk crop
    # - resize_cropped: Final size after resizing, or None to skip
    crop_config = {
        'downsample_resolution': 256,
        'crop_mode': 'disk',
        'resize_cropped': None
    }

    # -------------------------------------------------------------------------
    # Setup Paths
    # -------------------------------------------------------------------------

    base_dir = Path(os.getcwd()).parent
    paths = {'raw': base_dir / 'data' / 'unprocessed_images' / 'SDO',
             'preprocessed': base_dir / 'data' / 'preprocessed_images' / 'deep_learning',
             'cropped': base_dir / 'data' / 'preprocessed_images' / 'full_disk_cropped',  # 'dl_cropped',
             'config': base_dir / 'data' / 'configuration_data'}

    # -------------------------------------------------------------------------
    # Run Preprocessing
    # -------------------------------------------------------------------------

    print(f"Preprocessing channels: {channels}")
    print(f"Date range: {start_date} to {end_date}")

    preprocessor = SolarImagePreprocessor(
        channels=channels,
        paths=paths,
        use_gpu=use_gpu
    )
    preprocessor.run(
        start=start_date,
        end=end_date,
        load_preprocessing_fails=False,
        overwrite_existing=False
    )

    # -------------------------------------------------------------------------
    # Run Cropping
    # -------------------------------------------------------------------------

    print("\nStarting cropping...")
    cropper = ImageCropper(
        channels=channels,
        paths=paths,
        crop_config=crop_config
    )
    cropper.run(start=start_date, end=end_date)


def preprocess_single_channel(
    channel: str,
    start_date: datetime,
    end_date: datetime,
    base_dir: Path = None,
    use_gpu: bool = False
) -> None:
    """
    Preprocess a single channel.

    This is a convenience function for processing a single channel,
    useful for running channels in parallel on different machines.

    Parameters
    ----------
    channel : str
        Channel to process (e.g., 'aia_171', 'hmi').
    start_date : datetime
        Start date for preprocessing.
    end_date : datetime
        End date for preprocessing.
    base_dir : Path, optional
        Base directory for data. Defaults to project root.
    use_gpu : bool, optional
        Whether to use GPU for AIA deconvolution (default: False).

    Example
    -------
    >>> preprocess_single_channel(
    ...     channel='aia_171',
    ...     start_date=datetime(2020, 1, 1),
    ...     end_date=datetime(2020, 12, 31),
    ...     use_gpu=True
    ... )
    """
    if base_dir is None:
        base_dir = Path(os.getcwd()).parent

    paths = {'raw': base_dir / 'data' / 'unprocessed_images' / 'SDO',
             'preprocessed': base_dir / 'data' / 'preprocessed_images' / 'deep_learning',
             'cropped': base_dir / 'data' / 'preprocessed_images' / 'full_disk_cropped',#'dl_cropped',
             'config': base_dir / 'data' / 'configuration_data'}

    preprocessor = SolarImagePreprocessor(
        channels=[channel],
        paths=paths,
        use_gpu=use_gpu
    )
    preprocessor.run(start=start_date, end=end_date)


def crop_single_channel(
    channel: str,
    start_date: datetime,
    end_date: datetime,
    base_dir: Path = None,
    downsample_resolution: int = 256,
    crop_mode: str = 'disk',
    resize_cropped: int = None
) -> None:
    """
    Crop a single channel.

    This is a convenience function for cropping a single channel,
    useful for running channels in parallel on different machines.

    Parameters
    ----------
    channel : str
        Channel to process (e.g., 'aia_171', 'hmi').
    start_date : datetime
        Start date for cropping.
    end_date : datetime
        End date for cropping.
    base_dir : Path, optional
        Base directory for data. Defaults to project root.
    downsample_resolution : int, optional
        Target resolution after downsampling (default: 256).
    crop_mode : str, optional
        Crop mode: integer for fixed size or 'disk' (default: 'disk').
    resize_cropped : int, optional
        Final size after resizing, or None to skip (default: None).

    Example
    -------
    >>> crop_single_channel(
    ...     channel='aia_171',
    ...     start_date=datetime(2020, 1, 1),
    ...     end_date=datetime(2020, 12, 31),
    ...     crop_mode='disk'
    ... )
    """
    if base_dir is None:
        base_dir = Path(os.getcwd()).parent

    paths = {'raw': base_dir / 'data' / 'unprocessed_images' / 'SDO',
             'preprocessed': base_dir / 'data' / 'preprocessed_images' / 'deep_learning',
             'cropped': base_dir / 'data' / 'preprocessed_images' / 'full_disk_cropped',#'dl_cropped',
             'config': base_dir / 'data' / 'configuration_data'}

    crop_config = {
        'downsample_resolution': downsample_resolution,
        'crop_mode': crop_mode,
        'resize_cropped': resize_cropped
    }

    cropper = ImageCropper(
        channels=[channel],
        paths=paths,
        crop_config=crop_config
    )
    cropper.run(start=start_date, end=end_date)


if __name__ == "__main__":
    print('Starting preprocessing script')
    main()
    print('Finished preprocessing script')
