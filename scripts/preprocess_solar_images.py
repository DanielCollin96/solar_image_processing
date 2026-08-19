import argparse
from pathlib import Path

from solar_image_processing.utils.pipeline_config import PipelineConfig
from solar_image_processing.preprocessing.solar_image_preprocessor import SolarImagePreprocessor

def main():
    ap = argparse.ArgumentParser(description='Unified SDO/AIA preprocessing.')
    ap.add_argument('--mode', choices=['catch-up', 'incremental', 'realtime'], default='realtime')
    ap.add_argument('--months', type=int, default=2, help='lookback window (incremental/realtime)')
    ap.add_argument('--all', action='store_true', help='ignore the window; process full range')
    ap.add_argument('--retry-fails', action='store_true')
    ap.add_argument('--config', default=None)
    args = ap.parse_args()
    
    cfg_path = Path(args.config) if args.config else Path(__file__).resolve().parent.parent / 'configs' / 'pipeline_config.yaml'
    config = PipelineConfig(cfg_path)

    source = 'quicklook' if args.mode == 'realtime' else 'lev1'
    months = None if (args.mode == 'catch-up' or args.all) else args.months

    pre = SolarImagePreprocessor(config, source=source, months=months)
    pre.run()

if __name__ == '__main__':
    main()
