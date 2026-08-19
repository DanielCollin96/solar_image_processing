import argparse
from datetime import datetime, timedelta
from pathlib import Path

from solar_image_processing.cropping.solar_image_cropper import ImageCropper
from solar_image_processing.utils.pipeline_config import PipelineConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description='Crop preprocessed SDO solar images into a day-wise tree.'
    )
    parser.add_argument('--mode', choices=['catchup', 'realtime'], default='catchup')
    parser.add_argument('--backfill', action='store_true',
                        help='Process every month present on disk, ignoring the date range.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Report what would be cropped; write nothing.')
    parser.add_argument('--days-back', type=int, default=3,
                        help='Realtime only: days back from now to scan.')
    parser.add_argument('--start', type=str, default=None)
    parser.add_argument('--end', type=str, default=None)
    parser.add_argument('--config', type=Path, default=None)
    return parser.parse_args()


def parse_date(value):
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d'):
        try:
            return datetime.strptime(value.strip(), fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date '{value}'. Use YYYY-MM-DD or 'YYYY-MM-DD HH:MM:SS'.")


def resolve_config_path(explicit):
    if explicit is not None:
        return explicit.resolve()
    # Walk upward until configs/pipeline_config.yaml turns up, so this works
    # regardless of the src/ nesting or the working directory.
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / 'configs' / 'pipeline_config.yaml'
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f'Could not find configs/pipeline_config.yaml above {here}')


def main():
    args = parse_args()
    config = PipelineConfig(resolve_config_path(args.config))

    start_date = parse_date(args.start) if args.start else None
    end_date = parse_date(args.end) if args.end else None

    if args.mode == 'realtime' and not args.backfill and start_date is None:
        end_date = end_date or datetime.utcnow()
        start_date = end_date - timedelta(days=args.days_back)

    cropper = ImageCropper(
        config,
        mode=args.mode,
        start_date=start_date,
        end_date=end_date,
        backfill=args.backfill,
        dry_run=args.dry_run,
    )
    cropper.run()


if __name__ == '__main__':
    print('Starting cropping script')
    main()
    print('Finished cropping script')
