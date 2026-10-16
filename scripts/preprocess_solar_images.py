import argparse
from pathlib import Path

from solar_image_processing.utils.pipeline_config import PipelineConfig
from solar_image_processing.preprocessing.solar_image_preprocessor import SolarImagePreprocessor


def main():
    ap = argparse.ArgumentParser(description='Preprocess SDO/AIA images from JSOC or SIDC.')
    ap.add_argument('--source', choices=['jsoc', 'sidc'], required=True,
                    help='jsoc = JSOC lev1, sidc = SIDC quicklook.')
    ap.add_argument('--days', type=int, default=None,
                    help='Process the last N days up to now (default: 14).')
    ap.add_argument('--start', default=None,
                    help="Start of an explicit range, YYYY-MM-DD or 'YYYY-MM-DD HH:MM:SS'. Needs --end.")
    ap.add_argument('--end', default=None,
                    help='End of an explicit range (a date alone includes that whole day). Needs --start.')
    ap.add_argument('--all', action='store_true',
                    help='Process start_date -> end_date from the config.')
    ap.add_argument('--config', type=Path, default=None,
                    help='Config file (default: configs/pipeline_config.yaml).')
    args = ap.parse_args()

    cfg_path = args.config or Path(__file__).resolve().parent.parent / 'configs' / 'pipeline_config.yaml'
    config = PipelineConfig(cfg_path)

    try:
        start, end = config.resolve_window(args.days, args.start, args.end, args.all)
    except ValueError as e:
        ap.error(str(e))

    print(f'Source: {args.source}   window: {start:%Y-%m-%d %H:%M} -> {end:%Y-%m-%d %H:%M} (UTC)')
    SolarImagePreprocessor(config, source=args.source, start_date=start, end_date=end).run()


if __name__ == '__main__':
    main()
