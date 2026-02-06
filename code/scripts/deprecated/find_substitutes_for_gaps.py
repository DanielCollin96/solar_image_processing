from __future__ import annotations

import os
import pickle
import time
from copy import copy
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Union
from solar_image_downloader import SolarImageDownloader, create_aia_config, create_hmi_config, get_data_paths
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

import jsoc_download as jd
from utils import check_file_quality, find_missing_preprocessed_dates, read_file_name

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
    start_date = datetime(2026, 1, 1)
    channel = 'aia_171'  # Options: 'aia_171', 'aia_193', 'aia_211', 'hmi'
    rebin_factor = 4  # 4 for 1024x1024 images, 16 for 256x256 images


    # Email addresses for JSOC download
    # (rotate emails to avoid JSOC rate limits)
    email_address = 'collin@gfz.de'

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
    base_dir = Path(os.path.dirname(os.getcwd()))
    path_downloaded, path_preprocessed = get_data_paths(base_dir, config)

    # Ensure directories exist
    path_downloaded.mkdir(parents=True, exist_ok=True)
    path_preprocessed.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Run Downloader
    # -------------------------------------------------------------------------
    downloader = SolarImageDownloader(config, path_downloaded, path_preprocessed)
    downloader.download_images_hourly_cadence(start_date)

    # check if the download of missing preprocessed images is necessary, or if I acutally just want to first try again download the actual image and then adjacent images of missing downloaded

    # Download missing preprocessed images (main operation)
    downloader.download_missing_preprocessed_images(start_date)

    # Alternative operations (uncomment as needed):
    # downloader.substitute_manager.collect_all_substitute_dates(
    #     datetime(2010, 5, 1), datetime(2024, 7, 31)
    # )


if __name__ == '__main__':
    main()