from pathlib import Path

from solar_image_processing.downloading.solar_image_downloader import SolarImageDownloader
from solar_image_processing.utils.pipeline_config import PipelineConfig


def main() -> None:
    """
    Download SDO solar images according to the pipeline configuration.

    Reads all settings from ``configs/pipeline_config.yaml``, then runs the
    JSOC downloader for the configured channels at hourly cadence over the
    configured date range.

    Notes
    -----
    Edit ``configs/pipeline_config.yaml`` to change the channels, date range,
    email address, rebin factor, or output paths before running this script.
    """
    config_path = Path.cwd().parent / 'configs' / 'pipeline_config.yaml'
    config = PipelineConfig(config_path)

    downloader = SolarImageDownloader(config)
    downloader.download_images_hourly_cadence()


if __name__ == '__main__':
    print('Starting download script')
    main()
    print('Finished download script')
