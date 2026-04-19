"""
preprocess_realtime.py
======================
Preprocess AIA FITS files from the real-time data folder.

Handles both JSOC Level 1 and SIDC quicklook (Level 1.5) files.
The existing AIAPreprocessor (now with LVL_NUM-aware preprocess()) does
all the heavy lifting.

Outputs go to: data/preprocessed_images/output_realtime/YYYY/MM/
Filenames:     171_2026-03-18_23:59_lev1.npy   (and _lev1_meta.pickle)
               171_2026-03-18_15:48_ql.npy     (and _ql_meta.pickle)

Usage
-----
    cd /export/mp5/solardl/solar_image_processing/src
    uv run python -m solar_image_processing.preprocessing.preprocess_realtime

Place this file at:
    src/solar_image_processing/preprocessing/preprocess_realtime.py
"""

import os
import traceback
from datetime import datetime
from pathlib import Path

import astropy.units as u
import sunpy.map
from sunpy.map import contains_full_disk

from solar_image_processing.utils.pipeline_config import PipelineConfig
from solar_image_processing.utils.helper_functions import (
    read_file_name,
    save_preprocessed_output,
    load_calibration_data,
)
from solar_image_processing.preprocessing.aia_preprocessor import AIAPreprocessor


def collect_fits_files(base_path: Path):
    """Walk a directory tree and return all (directory, filename) pairs for FITS files."""
    results = []
    for dirpath, _, filenames in os.walk(base_path):
        for f in sorted(filenames):
            if f.endswith('.fits'):
                results.append((Path(dirpath), f))
    return results


def main() -> None:
    # ------------------------------------------------------------------ #
    #  Load config
    # ------------------------------------------------------------------ #
    config_path = Path(__file__).resolve().parents[3] / 'configs' / 'pipeline_config.yaml'
    if not config_path.is_file():
        config_path = Path.cwd().parent / 'configs' / 'pipeline_config.yaml'

    print(f'Loading config from {config_path}')
    config = PipelineConfig(config_path)

    path_input = config.paths['realtime_input']
    path_output = config.paths['realtime_output']
    path_instrument = config.paths['instrument_data']
    preproc_cfg = config.preprocessing_config

    print(f'Input folder:  {path_input}')
    print(f'Output folder: {path_output}')
    path_output.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Discover FITS files
    # ------------------------------------------------------------------ #
    fits_files = collect_fits_files(path_input)
    if not fits_files:
        print('No FITS files found. Nothing to do.')
        return

    print(f'Found {len(fits_files)} FITS file(s)')

    # ------------------------------------------------------------------ #
    #  Cache calibration data per channel
    # ------------------------------------------------------------------ #
    calibration_cache: dict = {}

    # ------------------------------------------------------------------ #
    #  Process each file
    # ------------------------------------------------------------------ #
    success = 0
    skipped = 0
    failed = 0

    for directory, filename in fits_files:
        print(f'\n{"─"*60}')
        print(f'File: {filename}')

        # --- parse filename ---
        try:
            file_date, product, channel, source = read_file_name(
                filename, preprocessed=False
            )
        except (ValueError, Exception) as e:
            print(f'  SKIP (cannot parse filename): {e}')
            skipped += 1
            continue

        if product != 'aia':
            print(f'  SKIP (not AIA: product={product})')
            skipped += 1
            continue

        print(f'  Date={file_date}, Channel={channel}, Source={source}')

        # --- load the FITS file ---
        fits_path = directory / filename
        try:
            aia_map = sunpy.map.Map(str(fits_path))
        except Exception:
            print(f'  SKIP (cannot read FITS)')
            skipped += 1
            continue

        # --- quality gate ---
        quality = aia_map.meta.get('QUALITY', 0)
        if quality != 0 and source != 'quicklook':
            print(f'  SKIP (QUALITY={quality})')
            skipped += 1
            continue

        if not contains_full_disk(aia_map):
            print(f'  SKIP (not full disk)')
            skipped += 1
            continue

        # --- downsample 4096 → 1024 before handing to preprocessor ---
        if aia_map.data.shape[0] == 4096:
            aia_map = aia_map.resample([1024, 1024] * u.pixel)

        # --- load calibration data (cached per channel) ---
        if channel not in calibration_cache:
            print(f'  Loading calibration data for channel {channel} ...')
            month_dt = datetime(file_date.year, file_date.month, 1)
            try:
                psf, corr_table, pointing_table = load_calibration_data(
                    path_instrument, channel, month_dt
                )
            except Exception as e:
                print(f'  WARNING: Full calibration load failed: {e}')
                print(f'  Trying without pointing table (OK for quicklook)...')
                try:
                    psf, corr_table, _ = load_calibration_data(
                        path_instrument, channel, month=None
                    )
                    pointing_table = None
                except Exception as e2:
                    print(f'  FAIL: Cannot load PSF/correction table: {e2}')
                    failed += 1
                    continue
            calibration_cache[channel] = (psf, corr_table, pointing_table)

        psf, corr_table, pointing_table = calibration_cache[channel]

        # --- build preprocessor and run ---
        preprocessor = AIAPreprocessor(
            pointing_table=pointing_table,
            point_spread_function=psf,
            correction_table=corr_table,
            config=preproc_cfg,
        )

        try:
            preprocessed_image, meta_info = preprocessor.preprocess(
                aia_map, map_date=file_date, target_date=file_date,
            )
        except Exception:
            print(f'  FAIL: preprocessing error')
            traceback.print_exc()
            failed += 1
            continue

        # --- save output ---
        sub_dir = path_output / file_date.strftime('%Y') / file_date.strftime('%m')
        try:
            save_preprocessed_output(
                sub_dir, channel, file_date,
                preprocessed_image, meta_info,
                source=source,
            )
            success += 1
        except Exception:
            print(f'  FAIL: could not save output')
            traceback.print_exc()
            failed += 1

    # ------------------------------------------------------------------ #
    #  Summary
    # ------------------------------------------------------------------ #
    print(f'\n{"="*60}')
    print(f'Done.  Success={success}  Skipped={skipped}  Failed={failed}')
    print(f'{"="*60}')


if __name__ == '__main__':
    print('Starting real-time preprocessing script')
    main()
    print('Finished real-time preprocessing script')
