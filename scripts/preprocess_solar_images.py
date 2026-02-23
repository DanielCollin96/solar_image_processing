from pathlib import Path

from solar_image_processing.preprocessing.solar_image_preprocessor import SolarImagePreprocessor
from solar_image_processing.utils.pipeline_config import PipelineConfig


def main() -> None:
    """
    Preprocess SDO solar images according to the pipeline configuration.

    Reads all settings from ``configs/pipeline_config.yaml``, then runs
    :class:`~solar_image_processing.preprocessing.solar_image_preprocessor.SolarImagePreprocessor`
    for all configured channels over the configured date range.

    Notes
    -----
    Edit ``configs/pipeline_config.yaml`` to change the channels, date range,
    GPU usage, differential rotation setting, or paths before running this
    script.
    """
    config_path = Path.cwd().parent / 'configs' / 'pipeline_config.yaml'
    config = PipelineConfig(config_path)

    preprocessor = SolarImagePreprocessor(config)
    preprocessor.run()


if __name__ == '__main__':
    print('Starting preprocessing script')
    main()
    print('Finished preprocessing script')
