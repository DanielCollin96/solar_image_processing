"""
Integration tests for the solar image processing pipeline.

Each test stage (download, preprocess, crop) runs the corresponding
pipeline class against a small reference dataset and compares the
output to stored reference files.
"""

import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import sunpy.map

from solar_image_processing.cropping.solar_image_cropper import ImageCropper
from solar_image_processing.downloading.solar_image_downloader import SolarImageDownloader
from solar_image_processing.preprocessing.solar_image_preprocessor import SolarImagePreprocessor
from solar_image_processing.utils.pipeline_config import PipelineConfig


def test_pipeline(
    downloading: bool = True,
    preprocessing: bool = True,
    cropping: bool = True,
) -> None:
    """
    Run end-to-end pipeline tests for all configured stages.

    Downloads are skipped by default because they require JSOC access and
    take significant time. Preprocessing and cropping run against a small
    reference dataset stored under ``tests/data/reference/``.

    Parameters
    ----------
    downloading : bool, optional
        If ``True``, test the JSOC download stage. Default is ``False``.
    preprocessing : bool, optional
        If ``True``, test the preprocessing stage. Default is ``True``.
    cropping : bool, optional
        If ``True``, test the cropping stage. Default is ``True``.
    """
    config_path = Path.cwd().parent / 'configs' / 'pipeline_config.yaml'
    config = PipelineConfig(config_path)

    test_data_dir = config.base_dir / 'tests' / 'data'
    tmp_dir = test_data_dir / 'tmp'
    tmp_dir.mkdir(exist_ok=True)

    if downloading:
        config.download_config['rebin_factor'] = 4
        config.paths['unprocessed'] = tmp_dir / 'data' / 'unprocessed_images' / 'SDO'

        test_channel_download('aia_171', datetime(2011, 1, 1, 0), config)
        test_channel_download('aia_193', datetime(2016, 1, 1, 0), config)
        test_channel_download('aia_211', datetime(2021, 1, 1, 0), config)
        test_channel_download('hmi', datetime(2015, 1, 1, 0), config)

    if preprocessing:
        config.preprocessing_config['differential_rotation'] = False
        config.paths['unprocessed'] = (
            config.base_dir / 'tests' / 'data' / 'reference' / 'unprocessed_images' / 'SDO'
        )
        config.paths['preprocessed'] = tmp_dir / 'data' / 'preprocessed_images' / 'deep_learning'
        config.paths['instrument_data'] = (
            config.base_dir / 'tests' / 'data' / 'reference' / 'instrument_data'
        )

        test_preprocessing('aia_171', datetime(2011, 1, 1, 0), config)
        test_preprocessing('aia_193', datetime(2016, 1, 1, 0), config)
        test_preprocessing('aia_211', datetime(2021, 1, 1, 0), config)
        test_preprocessing('hmi', datetime(2015, 1, 1, 0), config)

    if cropping:
        config.paths['preprocessed'] = (
            config.base_dir / 'tests' / 'data' / 'reference' / 'preprocessed_images' / 'deep_learning'
        )
        # Test full-disk crop at 256 px
        config.paths['cropped'] = tmp_dir / 'data' / 'preprocessed_images' / 'full_disk_cropped'
        config.cropping_config['downsample_resolution'] = 256
        config.cropping_config['crop_mode'] = 'disk'
        config.cropping_config['crop_pixels'] = None
        config.cropping_config['resize_cropped'] = None

        test_cropping('aia_193', datetime(2016, 1, 1, 0), config)

        # Test square crop at 512 px downsampled and resized to 224 px
        config.paths['cropped'] = tmp_dir / 'data' / 'preprocessed_images' / 'sws_prediction'
        config.cropping_config['downsample_resolution'] = 512
        config.cropping_config['crop_mode'] = 'square'
        config.cropping_config['crop_pixels'] = 300
        config.cropping_config['resize_cropped'] = 224

        test_cropping('aia_211', datetime(2021, 1, 1, 0), config)
        test_cropping('hmi', datetime(2015, 1, 1, 0), config)

    shutil.rmtree(tmp_dir)


def test_channel_download(
    channel: str,
    date: datetime,
    config: PipelineConfig,
) -> None:
    """
    Download one hour of data for a single channel and validate the output.

    Checks that exactly one FITS file and one metadata file are saved, then
    compares the downloaded image data against a stored reference map.

    Parameters
    ----------
    channel : str
        Channel identifier (e.g. ``'aia_171'``, ``'hmi'``).
    date : datetime
        Start of the one-hour download window.
    config : PipelineConfig
        Pipeline configuration (modified in place for this test).
    """
    config.channels = [channel]
    config.start_date = date
    config.end_date = date + timedelta(hours=1)

    downloader = SolarImageDownloader(config)
    downloader.download_images_hourly_cadence()

    if channel[:3] == 'aia':
        channel_path = config.paths['unprocessed'] / 'AIA' / channel[-3:]
    elif channel[:3] == 'hmi':
        channel_path = config.paths['unprocessed'] / 'HMI' / 'magnetogram'

    path_downloaded = channel_path / date.strftime('%Y/%m')
    file_list = os.listdir(path_downloaded)

    # Expect exactly one FITS file and one metadata pickle
    assert len(file_list) == 2

    # os.listdir order is arbitrary; identify the FITS file by prefix/suffix
    if file_list[0].startswith('meta_data'):
        assert file_list[1].endswith('.fits')
        downloaded_file = file_list[1]
    else:
        assert file_list[0].endswith('.fits')
        downloaded_file = file_list[0]

    downloaded_image = sunpy.map.Map(path_downloaded / downloaded_file)

    # Reference path mirrors the instrument subdirectory structure
    reference_path = (
        config.base_dir / 'tests' / 'data' / 'reference' / 'unprocessed_images' / 'SDO'
        / channel_path.parts[-2] / channel_path.parts[-1]
        / date.strftime('%Y/%m') / downloaded_file
    )
    reference = sunpy.map.Map(reference_path)

    assert np.allclose(np.nan_to_num(downloaded_image.data), np.nan_to_num(reference.data))


def test_preprocessing(
    channel: str,
    date: datetime,
    config: PipelineConfig,
) -> None:
    """
    Preprocess one hour of data for a single channel and validate the output.

    Checks that exactly one ``.npy`` image and one metadata pickle are saved,
    then compares the preprocessed array against a stored reference.

    Parameters
    ----------
    channel : str
        Channel identifier (e.g. ``'aia_171'``, ``'hmi'``).
    date : datetime
        Start of the one-hour preprocessing window.
    config : PipelineConfig
        Pipeline configuration (modified in place for this test).
    """
    config.channels = [channel]
    config.start_date = date
    config.end_date = date + timedelta(hours=1)
    config.preprocessing_config['overwrite_existing'] = True

    preprocessor = SolarImagePreprocessor(config)
    preprocessor.run()

    path_preprocessed = config.paths['preprocessed'] / channel / date.strftime('%Y/%m')
    file_list = os.listdir(path_preprocessed)
    file_list.remove('preprocessing_fails.csv')

    # Expect exactly one .npy image and one metadata pickle
    assert len(file_list) == 2

    # os.listdir order is arbitrary; identify the .npy file by suffix
    if file_list[0].endswith('meta.pickle'):
        assert file_list[1].endswith('.npy')
        preprocessed_file = file_list[1]
    else:
        assert file_list[0].endswith('.npy')
        preprocessed_file = file_list[0]

    preprocessed_image = np.load(path_preprocessed / preprocessed_file)

    reference_path = (
        config.base_dir / 'tests' / 'data' / 'reference' / 'preprocessed_images' / 'deep_learning'
        / channel / date.strftime('%Y/%m') / preprocessed_file
    )
    reference_image = np.load(reference_path)

    assert np.allclose(preprocessed_image, reference_image)


def test_cropping(
    channel: str,
    date: datetime,
    config: PipelineConfig,
) -> None:
    """
    Crop one preprocessed image for a single channel and validate the output.

    Checks that exactly one cropped ``.npy`` file is saved, then compares
    the cropped array against a stored reference.

    Parameters
    ----------
    channel : str
        Channel identifier (e.g. ``'aia_171'``, ``'hmi'``).
    date : datetime
        Date of the image to crop.
    config : PipelineConfig
        Pipeline configuration (modified in place for this test).
    """
    config.channels = [channel]
    config.start_date = date
    config.end_date = date + timedelta(hours=1)

    cropper = ImageCropper(config)
    cropper.run()

    path_cropped = config.paths['cropped'] / channel / date.strftime('%Y/%m')
    file_list = os.listdir(path_cropped)

    assert len(file_list) == 1
    cropped_file = file_list[0]

    cropped_image = np.load(path_cropped / cropped_file)

    # Last path component identifies the crop variant (e.g. 'full_disk_cropped')
    crop_variant = config.paths['cropped'].parts[-1]
    reference_path = (
        config.base_dir / 'tests' / 'data' / 'reference' / 'preprocessed_images'
        / crop_variant / channel / date.strftime('%Y/%m') / cropped_file
    )
    reference_image = np.load(reference_path)

    assert np.allclose(cropped_image, reference_image)


if __name__ == '__main__':
    print('Start testing pipeline')
    test_pipeline()
    print('Finished testing pipeline successfully')
