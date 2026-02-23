from pathlib import Path

from solar_image_processing.cropping.solar_image_cropper import ImageCropper
from solar_image_processing.utils.pipeline_config import PipelineConfig


def main() -> None:
    """
    Crop preprocessed SDO solar images according to the pipeline configuration.

    Reads all settings from ``configs/pipeline_config.yaml``, then runs
    :class:`~solar_image_processing.cropping.solar_image_cropper.ImageCropper`
    for all configured channels over the configured date range.

    Notes
    -----
    Edit ``configs/pipeline_config.yaml`` to change the channels, date range,
    downsample resolution, crop mode, crop size, final resize target, or paths
    before running this script.
    """
    config_path = Path.cwd().parent / 'configs' / 'pipeline_config.yaml'
    config = PipelineConfig(config_path)

    cropper = ImageCropper(config)
    cropper.run()


if __name__ == '__main__':
    print('Starting cropping script')
    main()
    print('Finished cropping script')
