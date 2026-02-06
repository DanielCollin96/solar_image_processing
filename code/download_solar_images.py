#!/usr/bin/env python
"""
Solar Image Download Script.

This script provides the main entry point for downloading SDO (Solar Dynamics
Observatory) solar images from JSOC. It demonstrates typical usage of the
downloader and can be customized by modifying the configuration parameters.

Example Usage:
    python download_solar_images.py

See Also:
    src/solar_data/solar_image_downloader.py : Module containing downloader classes
    preprocess_solar_images.py : Companion script for preprocessing images
"""

import os
import sys
from datetime import datetime
from pathlib import Path


from solar_image_downloader import (
    SolarImageDownloader,
    create_aia_config,
    create_hmi_config,
    get_data_paths,
)


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """
    Main entry point for running the solar image downloader.

    This function demonstrates typical usage of the downloader and can
    be customized by modifying the configuration parameters below.
    """
    # -------------------------------------------------------------------------
    # Configuration Parameters
    # -------------------------------------------------------------------------

    # Start date for downloading
    start_date = datetime(2026, 1, 1)

    # Channel to download
    # Options: 'aia_171', 'aia_193', 'aia_211', 'hmi'
    channel = 'aia_171'

    # Rebin factor for image size reduction on JSOC server
    # 4 = 1024x1024 images, 16 = 256x256 images
    rebin_factor = 4

    # Email address registered with JSOC for data requests
    # Register at: http://jsoc.stanford.edu/ajax/exportdata.html
    email_address = 'your_email@example.com'

    # -------------------------------------------------------------------------
    # Create Configuration
    # -------------------------------------------------------------------------

    if channel.startswith('aia_'):
        wavelength = int(channel.split('_')[1])
        config = create_aia_config(email_address, wavelength, rebin_factor)
    elif channel == 'hmi':
        config = create_hmi_config(email_address, rebin_factor)
    else:
        raise ValueError(f"Unknown channel: {channel}")

    # -------------------------------------------------------------------------
    # Setup Paths
    # -------------------------------------------------------------------------

    base_dir =  Path(os.getcwd()).parent
    path_downloaded, path_preprocessed = get_data_paths(base_dir, config)

    # Ensure directories exist
    path_downloaded.mkdir(parents=True, exist_ok=True)
    path_preprocessed.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Run Downloader
    # -------------------------------------------------------------------------

    print(f"Downloading {channel} images starting from {start_date}")
    print(f"Output directory: {path_downloaded}")

    downloader = SolarImageDownloader(config, path_downloaded, path_preprocessed)
    downloader.download_images_hourly_cadence(start_date)


def download_channel(
    channel: str,
    start_date: datetime,
    email_address: str,
    base_dir: Path = None,
    rebin_factor: int = 4
) -> None:
    """
    Download images for a specific channel.

    This is a convenience function for downloading a single channel,
    useful for running multiple channels in parallel or from a script.

    Parameters
    ----------
    channel : str
        Channel to download (e.g., 'aia_171', 'aia_193', 'aia_211', 'hmi').
    start_date : datetime
        Date to start downloading from.
    email_address : str
        JSOC-registered email address.
    base_dir : Path, optional
        Base directory for data storage. Defaults to project root.
    rebin_factor : int, optional
        Rebinning factor (default: 4 for 1024x1024 images).

    Example
    -------
    >>> download_channel(
    ...     channel='aia_171',
    ...     start_date=datetime(2020, 1, 1),
    ...     email_address='user@example.com'
    ... )
    """
    if base_dir is None:
        base_dir = Path(os.getcwd()).parent

    if channel.startswith('aia_'):
        wavelength = int(channel.split('_')[1])
        config = create_aia_config(email_address, wavelength, rebin_factor)
    elif channel == 'hmi':
        config = create_hmi_config(email_address, rebin_factor)
    else:
        raise ValueError(f"Unknown channel: {channel}")

    path_downloaded, path_preprocessed = get_data_paths(base_dir, config)
    path_downloaded.mkdir(parents=True, exist_ok=True)
    path_preprocessed.mkdir(parents=True, exist_ok=True)

    downloader = SolarImageDownloader(config, path_downloaded, path_preprocessed)
    downloader.download_images_hourly_cadence(start_date)


if __name__ == '__main__':
    print('Starting download script')
    main()
    print('Finished download script')
