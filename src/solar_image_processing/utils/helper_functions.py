
import os
import pickle
from copy import copy
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple


import astropy.units as u
import numpy as np
import pandas as pd
import sunpy.map
from aiapy.calibrate.util import get_correction_table, get_pointing_table
from aiapy.psf import psf as calculate_psf
from astropy.time import Time
from dateutil.relativedelta import relativedelta
from joblib import Parallel, delayed, cpu_count
from sunpy.map import contains_full_disk

from solar_image_processing.psf_deconvolution.rebin_psf import rebin_psf

BENIGN_QUALITY_BITS = (1 << 30) | (1 << 2) | (1 << 21)   # quicklook label, ASD_REC, AIAGP6

def quality_ok(quality, source='lev1'):
    """QL: keep clean + benign (bits 2/21). lev1: strict QUALITY==0 (batch unchanged)."""
    q = int(quality) & 0xFFFFFFFF
    if source == 'quicklook':
        return (q & ~BENIGN_QUALITY_BITS) == 0
    return q == 0

def create_folders_for_preprocessed_images(
    start: datetime,
    end: datetime,
    path_to_preprocessed: str,
) -> None:
    """
    Create year/month subdirectories for preprocessed image output.

    Parameters
    ----------
    start : datetime
        Start of the date range.
    end : datetime
        End of the date range (exclusive).
    path_to_preprocessed : str
        Base directory path (must end with a path separator).
    """
    if not os.path.isdir(path_to_preprocessed):
        os.mkdir(path_to_preprocessed)

    current_month = copy(start)
    while current_month < end:
        if not os.path.isdir(path_to_preprocessed + current_month.strftime('%Y')):
            os.mkdir(path_to_preprocessed + current_month.strftime('%Y'))

        if not os.path.isdir(path_to_preprocessed + current_month.strftime('%Y/%m')):
            os.mkdir(path_to_preprocessed + current_month.strftime('%Y/%m'))

        current_month = current_month + relativedelta(months=1)


def find_missing_cropped_dates(
    month: datetime,
    path_to_cropped_files: Path,
    channel: str,
) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]:
    """
    Identify hourly target dates that have not yet been cropped.

    Parameters
    ----------
    month : datetime
        Month to check (day component is ignored).
    path_to_cropped_files : Path
        Directory containing cropped ``.npy`` files.
    channel : str
        Channel identifier (e.g. ``'aia_171'``, ``'hmi'``).

    Returns
    -------
    Tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]
        ``(missing_dates, existing_dates, target_dates)`` where
        ``target_dates`` is the full hourly grid for the month.
    """
    channel_str = channel.split('_')[1] if 'aia' in channel else channel

    # SDO science operations started on 2010-05-18
    if month.year == 2010 and month.month == 5:
        month_start = datetime(2010, 5, 18)
        month_end = datetime(2010, 5, 31, 23)
    else:
        month_start = datetime(month.year, month.month, 1, 0)
        month_end = month_start + relativedelta(months=1) - timedelta(hours=1)
    target_dates = pd.date_range(month_start, month_end, freq='1h')

    existing_cropped_dates = []
    if os.path.isdir(path_to_cropped_files):
        for day_name in sorted(os.listdir(path_to_cropped_files)):
            day_path = path_to_cropped_files / day_name
            if not day_path.is_dir():
                continue
            for file in sorted(os.listdir(day_path)):
                if file.endswith('.npy') and file[:3] == channel_str:
                    file_date, _, _, _ = read_file_name(file, preprocessed=True)
                    existing_cropped_dates.append(file_date)

    existing_cropped_dates = pd.DatetimeIndex(existing_cropped_dates)
    missing_cropped_dates = target_dates.difference(existing_cropped_dates)
    return missing_cropped_dates, existing_cropped_dates, target_dates


def find_missing_preprocessed_dates(
    month: datetime,
    path_to_preprocessed_files: Path,
    channel: str,
    overwrite_existing: bool = False,
) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """
    Identify hourly target dates that have not yet been preprocessed.

    Parameters
    ----------
    month : datetime
        Month to check (day component is ignored).
    path_to_preprocessed_files : Path
        Directory containing preprocessed ``.npy`` files.
    channel : str
        Channel identifier (e.g. ``'aia_171'``, ``'hmi'``).
    overwrite_existing : bool, optional
        If ``True``, treat all target dates as missing regardless of what
        exists on disk. Default is ``False``.

    Returns
    -------
    Tuple[pd.DatetimeIndex, pd.DatetimeIndex]
        ``(missing_dates, target_dates)`` where ``target_dates`` is the
        full hourly grid for the month.
    """
    channel_str = channel.split('_')[1] if 'aia' in channel else channel

    # SDO science operations started on 2010-05-18
    if month.year == 2010 and month.month == 5:
        month_start = datetime(2010, 5, 18)
        month_end = datetime(2010, 5, 31, 23)
    else:
        month_start = datetime(month.year, month.month, 1, 0)
        month_end = month_start + relativedelta(months=1) - timedelta(hours=1)
    target_dates = pd.date_range(month_start, month_end, freq='1h')

    if overwrite_existing:
        return target_dates, target_dates

    existing_preprocessed_dates = []
    if os.path.isdir(path_to_preprocessed_files):
        for day_name in sorted(os.listdir(path_to_preprocessed_files)):
            day_path = path_to_preprocessed_files / day_name
            if not day_path.is_dir():
                continue
            for file in sorted(os.listdir(day_path)):
                if file.endswith('.npy') and file[:3] == channel_str:
                    file_date, _, _, _ = read_file_name(file, preprocessed=True)
                    existing_preprocessed_dates.append(file_date)

    existing_preprocessed_dates = pd.DatetimeIndex(existing_preprocessed_dates)
    missing_preprocessed_dates = target_dates.difference(existing_preprocessed_dates)
    return missing_preprocessed_dates, target_dates


def load_existing_preprocessed_files(
    path_to_preprocessed_files: Path,
    channel: str,
) -> pd.Series:
    """
    Index the preprocessed files that exist for one month, keyed by target date.

    Walks the ``DD/`` subdirectories beneath a month-level directory and
    returns the filenames found, each prefixed with its day folder so that
    ``path_to_preprocessed_files / value`` resolves correctly.

    Counterpart to :func:`load_existing_raw_files`. Unlike
    :func:`load_existing_preprocessed_dates`, which returns dates only, this
    keeps the filename so callers never reconstruct one. That matters because
    the source tag (``_lev1`` / ``_ql``) cannot be derived from a date: which
    hours came from JSOC and which from the SIDC quicklook stream is recorded
    only in the filename on disk.

    Parameters
    ----------
    path_to_preprocessed_files : Path
        Month-level directory, e.g. ``.../uncropped/aia_171/2026/07``.
    channel : str
        Channel identifier, e.g. ``'aia_171'`` or ``'hmi'``.

    Returns
    -------
    pd.Series
        Index: target dates. Values: ``'DD/filename.npy'``, tag included.
        Empty Series with a DatetimeIndex if the directory does not exist.
    """
    channel_str = channel.split('_')[1] if 'aia' in channel else channel

    file_dates = []
    file_names = []

    if os.path.isdir(path_to_preprocessed_files):
        for day_name in sorted(os.listdir(path_to_preprocessed_files)):
            day_path = path_to_preprocessed_files / day_name
            if not day_path.is_dir():
                continue

            for file in sorted(os.listdir(day_path)):
                if not file.endswith('.npy'):
                    continue
                if file[:len(channel_str)] != channel_str:
                    continue

                file_date, _, _, _ = read_file_name(file, preprocessed=True)
                file_dates.append(file_date)
                # Keep the day folder so 'path / value' resolves in the
                # day-wise tree.
                file_names.append(f'{day_name}/{file}')

    index = pd.DatetimeIndex(file_dates)
    if len(index) == 0:
        return pd.Series(dtype='object', index=pd.DatetimeIndex([]))

    # One file per target date. If a lev1 and a quicklook file both exist for
    # the same hour, keep the first in sorted order ('_lev1' sorts before
    # '_ql') and say so rather than failing silently.
    duplicated = index.duplicated(keep='first')
    if duplicated.any():
        clashes = sorted(set(index[duplicated]))
        print(
            f'  WARNING: {len(clashes)} date(s) have more than one preprocessed '
            f'file in {path_to_preprocessed_files}; keeping the first of each. '
            f'First clash: {clashes[0]}'
        )

    return pd.Series(
        pd.Index(file_names)[~duplicated],
        index=index[~duplicated],
    ).sort_index()


def load_existing_raw_files(path_to_raw_files: Path) -> pd.Series:
    """
    Index the raw FITS files that exist for one month, keyed by observation date.

    Walks the ``DD/`` subdirectories beneath a month-level directory, mirroring
    the day-wise layout the downloader writes to. Counterpart to
    :func:`load_existing_preprocessed_files`.

    Parameters
    ----------
    path_to_raw_files : Path
        Month-level directory, e.g. ``.../AIA/171/2026/07``.

    Returns
    -------
    pd.Series
        Index: observation datetime. Values: ``'DD/filename.fits'``, so that
        ``path_to_raw_files / value`` resolves correctly.
        Empty Series with a DatetimeIndex if the directory does not exist.
    """
    file_dates = []
    file_names = []

    if os.path.isdir(path_to_raw_files):
        for day_name in sorted(os.listdir(path_to_raw_files)):
            day_path = path_to_raw_files / day_name
            if not day_path.is_dir():
                continue

            for file in sorted(os.listdir(day_path)):
                if not file.endswith('.fits'):
                    continue

                file_date, _, _, _ = read_file_name(file)
                file_dates.append(file_date)
                file_names.append(f'{day_name}/{file}')

    index = pd.DatetimeIndex(file_dates)
    if len(index) == 0:
        return pd.Series(dtype='object', index=pd.DatetimeIndex([]))

    # Keep only the first file for each observation time
    duplicated = index.duplicated(keep='first')
    return pd.Series(
        pd.Index(file_names)[~duplicated],
        index=index[~duplicated],
    ).sort_index()

def keep_hourly(raw_files):
    """Keep one frame per hour — the one nearest :00 — preserving its true obstime."""
    if len(raw_files) == 0:
        return raw_files
    idx = raw_files.index
    nearest_hour = idx.round('h')
    order = np.argsort(np.abs((idx - nearest_hour).total_seconds()))   # ← fix
    seen, keep = set(), []
    for i in order:
        h = nearest_hour[i]
        if h not in seen:
            seen.add(h); keep.append(idx[i])
    return raw_files.loc[sorted(keep)]

def find_available_months(path_to_channel: Path) -> list:
    """
    List the months that contain data beneath a channel directory.

    Used by the cropper's backfill mode, which processes everything present
    on disk rather than the date range in the configuration.

    Parameters
    ----------
    path_to_channel : Path
        Channel-level directory, e.g. ``.../uncropped/aia_171``.

    Returns
    -------
    list of datetime
        First-of-month datetimes, ascending. Empty if the directory does not
        exist or holds no ``YYYY/MM`` subdirectories.
    """
    months = []

    if not os.path.isdir(path_to_channel):
        return months

    for year_name in sorted(os.listdir(path_to_channel)):
        year_path = path_to_channel / year_name
        if not year_path.is_dir() or not year_name.isdigit():
            continue

        for month_name in sorted(os.listdir(year_path)):
            month_path = year_path / month_name
            if not month_path.is_dir() or not month_name.isdigit():
                continue

            months.append(datetime(int(year_name), int(month_name), 1))

    return sorted(months)

def load_calibration_data(
    path_to_config: Path,
    wl: str,
    month: Optional[datetime] = None,
) -> Tuple[np.ndarray, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load (or compute and cache) AIA calibration data for a given wavelength.

    Loads the PSF, degradation correction table, and — if ``month`` is
    given — the pointing table. Each item is read from a cached pickle if
    available; otherwise it is downloaded/computed and saved.

    Parameters
    ----------
    path_to_config : Path
        Directory where calibration pickle files are stored.
    wl : str
        AIA wavelength in Angstroms (e.g. ``'171'``).
    month : datetime, optional
        Month for which to load the pointing table. If ``None``, the
        pointing table is not loaded and ``None`` is returned for it.

    Returns
    -------
    Tuple[np.ndarray, pd.DataFrame, Optional[pd.DataFrame]]
        ``(psf_rebinned, correction_table, pointing_table)``
    """
    rebin_dimension = [1024, 1024]
    psf_rebinned_path = path_to_config / f'psf_{wl}_{rebin_dimension[0]}x{rebin_dimension[1]}.pickle'
    psf_path = path_to_config / f'psf_{wl}.pickle'

    print('Loading PSF.')
    try:
        with open(psf_rebinned_path, 'rb') as f:
            psf_rebinned = pickle.load(f)
    except Exception:
        # Rebinned PSF not cached; fall back to full-resolution PSF
        try:
            with open(psf_path, 'rb') as f:
                psf = pickle.load(f)
        except Exception:
            # No cached PSF; compute from scratch (slow without GPU)
            psf = calculate_psf(int(wl) * u.angstrom)
            with open(psf_path, 'wb') as f:
                pickle.dump(psf, f)

        psf_rebinned = rebin_psf(psf, rebin_dimension)
        with open(psf_rebinned_path, 'wb') as f:
            pickle.dump(psf_rebinned, f)

    print('Loading degradation correction table.')
    correction_table_path = path_to_config / 'degradation_correction_table.pickle'
    try:
        with open(correction_table_path, 'rb') as f:
            correction_table = pickle.load(f)
    except Exception:
        print('Did not find saved degradation table. Downloading it.')
        correction_table = get_correction_table("JSOC")
        with open(correction_table_path, 'wb') as f:
            pickle.dump(correction_table, f)

    if month is not None:
        print(f'Loading pointing table for {month.strftime("%Y%m")}.')
        pointing_table_path = path_to_config / f'pointing_table_{month.strftime("%Y%m")}.pickle'
        try:
            with open(pointing_table_path, 'rb') as f:
                pointing_table = pickle.load(f)
        except Exception:
            print('Did not find saved pointing table. Downloading it.')
            # Download ±1 month window to cover the full month
            time_range = (
                Time(month - timedelta(days=1)),
                Time(month + timedelta(days=32)),
            )
            pointing_table = get_pointing_table("JSOC", time_range=time_range)
            with open(pointing_table_path, 'wb') as f:
                pickle.dump(pointing_table, f)
    else:
        pointing_table = None

    return psf_rebinned, correction_table, pointing_table


def check_file_quality(
    files: List[str],
    path_to_downloaded: Path,
    source='lev1'
) -> Tuple[List[datetime], List[datetime]]:
    """
    Assess each FITS file and separate good from bad observations.

    Returns immediately after the first valid file is found (caller
    passes files sorted by temporal closeness).

    Parameters
    ----------
    files : List[str]
        Filenames to check, in priority order.
    path_to_downloaded : Path
        Directory containing the FITS files.
source='lev1'
    Returns
    -------
    Tuple[List[datetime], List[datetime]]
        ``(good_dates, bad_dates)`` where ``good_dates`` contains at most
        one entry (the first file passing all quality checks).
    """
    good_dates: List[datetime] = []
    bad_dates: List[datetime] = []

    for file in files:
        if not file.endswith('.fits'):
            continue

        file_date, _, _, _ = read_file_name(file)
        fits_file = path_to_downloaded / file

        try:
            smap = sunpy.map.Map(fits_file)
            reading_success = True
        except Exception:
            reading_success = False

        if reading_success:
            full_disk = contains_full_disk(smap)
            good_quality = quality_ok(smap.meta['QUALITY'], source)

            if full_disk and good_quality:
                good_dates.append(file_date)
                return good_dates, bad_dates  # First good file found; stop
            else:
                bad_dates.append(file_date)
        else:
            bad_dates.append(file_date)

    return good_dates, bad_dates


def read_file_name(
    file: str,
    preprocessed: bool = False,
) -> Tuple[datetime, str, str]:
    """
    Parse instrument, channel, and observation date from a filename.

    Parameters
    ----------
    file : str
        Filename of a FITS or ``.npy``/``.pickle`` file.
    preprocessed : bool, optional
        If ``True``, parse a preprocessed filename (``<channel>_<date>.npy``).
        If ``False``, parse a raw JSOC FITS filename. Default is ``False``.

    Returns
    -------
    Tuple[datetime, str, str]
        ``(file_date, product, channel)`` where ``product`` is ``'aia'`` or
        ``'hmi'`` and ``channel`` is the wavelength string (AIA) or ``''`` (HMI).
    """
    file = str(file)

    if preprocessed:
        if file[:3] in ('171', '193', '211'):
            product = 'aia'
            channel = file[:3]
        elif file[:3] == 'hmi':
            product = 'hmi'
            channel = ''

        if '.pickle' in file:
            date_str = file.split('.pickle')[0][4:]
        elif '.npy' in file:
            date_str = file.split('.npy')[0][4:]
        for suffix in ('_lev1', '_ql'):
            date_str = date_str.replace(suffix, '')
        file_date = datetime.strptime(date_str, '%Y-%m-%d_%H:%M')

    else:
        file = file.split('/')[-1]
        # --- Quicklook format ---
        # aia_quicklook.0171.20260318_154800.fits
        if file.startswith('aia_quicklook'):
            product = 'aia'
            parts = file.split('.')
            channel = str(int(parts[1]))
            datetime_str = parts[2]
            file_date = datetime.strptime(datetime_str, '%Y%m%d_%H%M%S')
            return file_date, product, channel, 'quicklook'

        # --- JSOC formats (existing) ---
        product = file[:3]
        if product == 'hmi':
            channel = ''
            date_str = file.split('720s.')[1][:8]
            time_str = file.split('_TAI')[0][-6:]
            # JSOC occasionally encodes seconds as 60; clamp to 59
            if int(time_str[-2:]) > 59:
                time_list = list(time_str)
                time_list[-2:] = '59'
                time_str = ''.join(time_list)
            file_date = datetime.strptime(date_str + '_' + time_str, '%Y%m%d_%H%M%S')

        elif product == 'aia':
            try:
                # Format 1: standard JSOC AIA filename
                channel = file.split('.image')[0][-3:]
                date_str = file.split('T')[0][-10:]
                time_str = file.split('T')[1][:6]
                if int(time_str[-2:]) > 59:
                    time_list = list(time_str)
                    time_list[-2:] = '59'
                    time_str = ''.join(time_list)
                file_date = datetime.strptime(date_str + '_' + time_str, '%Y-%m-%d_%H%M%S')
            except Exception:
                # Format 2: alternative JSOC AIA filename convention
                channel = file.split('lev1_')[1][:3]
                date_str = file.split('t')[0][-10:]
                time_str = file.split('t')[1][:8]
                if int(time_str[-2:]) > 59:
                    time_list = list(time_str)
                    time_list[-2:] = '59'
                    time_str = ''.join(time_list)
                file_date = datetime.strptime(date_str + '_' + time_str, '%Y_%m_%d_%H_%M_%S')

    return file_date, product, channel, 'lev1'


def check_completeness_of_preprocessed_images(
    files_to_exclude: pd.DataFrame,
    current_month: datetime,
    path_to_preprocessed_files: Path,
    channel: str,
) -> Tuple[bool, List[datetime]]:
    """
    Check whether all target dates for a month have been preprocessed.

    Dates are considered acceptable if they are either preprocessed or
    listed in ``files_to_exclude`` as bad/missing raw data.

    Parameters
    ----------
    files_to_exclude : pd.DataFrame
        DataFrame (indexed by date) with boolean columns ``'bad'`` and
        ``'missing_raw'`` identifying dates that cannot be preprocessed.
    current_month : datetime
        Month to validate.
    path_to_preprocessed_files : Path
        Directory containing preprocessed ``.npy`` files for this month.
    channel : str
        Channel identifier (e.g. ``'aia_171'``, ``'hmi'``).

    Returns
    -------
    Tuple[bool, List[datetime]]
        ``(all_complete, dates_to_check)`` where ``dates_to_check`` lists
        dates that are missing but not explained by ``files_to_exclude``.
    """
    missing_preprocessed_dates, _ = find_missing_preprocessed_dates(
        current_month, path_to_preprocessed_files, channel, overwrite_existing=False
    )

    dates_to_check = []
    for missing_date in missing_preprocessed_dates:
        # A missing date is acceptable if it is already flagged as bad or missing raw
        check_date = True
        if missing_date in files_to_exclude.index:
            if files_to_exclude.loc[missing_date, 'bad'] or files_to_exclude.loc[missing_date, 'missing_raw']:
                check_date = False
        if check_date:
            dates_to_check.append(missing_date)

    if len(dates_to_check) == 0:
        print(f'All possible dates for {current_month.strftime("%Y/%m")} successfully preprocessed.')
        all_successfully_preprocessed = True
    else:
        print('The following dates are downloaded and of good quality but have not been preprocessed successfully:')
        print(dates_to_check)
        all_successfully_preprocessed = False

    return all_successfully_preprocessed, dates_to_check


def find_substitute_file(
    missing_date: datetime,
    existing_raw_files: pd.Series,
    path_to_raw_files: Path,
    source='lev1',
) -> Tuple[datetime, Optional[str], bool, bool]:
    """
    Find the best available substitute FITS file for a missing target date.

    Searches within a ±24.5 h window around ``missing_date`` and selects
    the temporally closest file that passes quality checks.

    Parameters
    ----------
    missing_date : datetime
        Target date for which no preprocessed image exists.
    existing_raw_files : pd.Series
        Series with observation datetime as index and filename as values.
    path_to_raw_files : Path
        Directory containing the raw FITS files.

    Returns
    -------
    Tuple[datetime, Optional[str], bool, bool]
        ``(missing_date, best_filename, bad_raw, missing_raw)`` where
        ``best_filename`` is ``None`` if no usable file was found,
        ``bad_raw`` is ``True`` if candidates exist but all fail quality checks,
        and ``missing_raw`` is ``True`` if no candidates exist at all.
    """
    bad_raw_candidates = False
    missing_raw_map = False
    files_to_preprocess_name = None

    # Search window of ±24.5 h covers gap-filling from adjacent days
    window = timedelta(hours=24, minutes=30)
    mask = (
        (existing_raw_files.index > missing_date - window)
        & (existing_raw_files.index < missing_date + window)
    )

    if np.sum(mask) > 0:
        files_to_check = existing_raw_files.loc[mask]
        # Sort candidates by temporal distance so quality check stops at closest good file
        time_difference = np.abs(missing_date - files_to_check.index)
        sort_index = np.argsort(time_difference)
        files_to_check = files_to_check.iloc[sort_index]

        good_dates, _ = check_file_quality(list(files_to_check), path_to_raw_files, source)

        if len(good_dates) > 0:
            # Select the temporally closest good candidate
            time_differences = np.abs(missing_date - pd.DatetimeIndex(good_dates))
            best_candidate_date = good_dates[np.argmin(time_differences)]
            files_to_preprocess_name = existing_raw_files.loc[best_candidate_date]
        else:
            bad_raw_candidates = True
    else:
        missing_raw_map = True

    return missing_date, files_to_preprocess_name, bad_raw_candidates, missing_raw_map


def find_files_to_preprocess(
    missing_preprocessed_dates: pd.DatetimeIndex,
    existing_raw_files: pd.Series,
    path_to_raw_files: Path,
    source='lev1',
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Match missing target dates to the best available raw substitute files.

    Runs ``_find_substitute_file`` in parallel across all missing dates and
    returns a mapping of filenames to target dates, plus a table of dates
    that cannot be preprocessed.

    Parameters
    ----------
    missing_preprocessed_dates : pd.DatetimeIndex
        Target dates for which preprocessed files are absent.
    existing_raw_files : pd.Series
        Series with observation datetime as index and filename as values.
    path_to_raw_files : Path
        Directory containing the raw FITS files.

    Returns
    -------
    Tuple[pd.Series, pd.DataFrame]
        ``(files_to_preprocess, files_to_exclude)`` where
        ``files_to_preprocess`` has filenames as index and target dates as
        values, and ``files_to_exclude`` is a DataFrame with boolean columns
        ``'bad'`` and ``'missing_raw'`` for unresolvable dates.
    """
    n_cpus = cpu_count()
    print('Analysing files to find substitutes for missing dates.')
    print(f'Number of available CPUs: {n_cpus}')

    results = Parallel(n_jobs=n_cpus // 2)(
        delayed(find_substitute_file)(date, existing_raw_files, path_to_raw_files, source)
        for date in missing_preprocessed_dates
    )

    # Stack results into a DataFrame: index = target dates, columns = outcome
    results_arr = np.array(results, dtype=object)
    results_df = pd.DataFrame(
        results_arr[:, 1:],
        index=results_arr[:, 0],
        columns=['file_name', 'bad', 'missing_raw'],
    )

    # Separate dates with a valid substitute from those without
    nan_mask = pd.isna(results_df['file_name']).values
    valid_mask = ~nan_mask

    # Invert index/values: result has filename as index, target date as value
    files_to_preprocess = pd.Series(
        results_df.index[valid_mask],
        index=results_df['file_name'].values[valid_mask],
    )
    files_to_exclude = results_df.loc[nan_mask, ['bad', 'missing_raw']]

    return files_to_preprocess, files_to_exclude

def save_preprocessed_output(
    path_output: Path,
    channel: str,
    target_date: datetime,
    image: np.ndarray,
    metadata: dict,
    source: str = 'lev1',
) -> None:
    """
    Save a preprocessed image array and its metadata to disk.

    Parameters
    ----------
    path_output : Path
        Output directory.
    channel : str
        Channel identifier used in the filename.
    target_date : datetime
        Target date used in the filename.
    image : np.ndarray
        Preprocessed image array.
    metadata : dict
        Image metadata (saved as a pickle alongside the array).
    """
    tag = 'ql' if source == 'quicklook' else 'lev1'
    date_str = target_date.strftime('%Y-%m-%d_%H:%M')
    base_name = f'{channel}_{date_str}'

    # Write into day subfolder for day-wise output layout
    day_path = path_output / f'{target_date.day:02d}'
    day_path.mkdir(parents=True, exist_ok=True)

    np.save(day_path / f'{base_name}.npy', image)
    with open(day_path / f'{base_name}_meta.pickle', 'wb') as f:
        pickle.dump(metadata, f)

    print(f'Saved {base_name}.npy')
