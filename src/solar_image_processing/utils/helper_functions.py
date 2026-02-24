import os
import pickle
from copy import copy
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import aiapy.calibrate.utils
import astropy.units as u
import numpy as np
import pandas as pd
import sunpy.map
from aiapy.calibrate.utils import get_correction_table, get_pointing_table
from aiapy.psf import calculate_psf
from astropy.time import Time
from dateutil.relativedelta import relativedelta
from joblib import Parallel, delayed, cpu_count
from sunpy.map import contains_full_disk

from solar_image_processing.psf_deconvolution.rebin_psf import rebin_psf



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

    cropped_files = sorted(os.listdir(path_to_cropped_files))
    existing_cropped_dates = []
    for file in cropped_files:
        if file.endswith('.npy') and file[:3] == channel_str:
            file_date, _, _ = read_file_name(file, preprocessed=True)
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

    preprocessed_files = sorted(os.listdir(path_to_preprocessed_files))
    existing_preprocessed_dates = []
    for file in preprocessed_files:
        if file.endswith('.npy') and file[:3] == channel_str:
            file_date, _, _ = read_file_name(file, preprocessed=True)
            existing_preprocessed_dates.append(file_date)

    existing_preprocessed_dates = pd.DatetimeIndex(existing_preprocessed_dates)
    missing_preprocessed_dates = target_dates.difference(existing_preprocessed_dates)
    return missing_preprocessed_dates, target_dates


def load_existing_preprocessed_dates(
    path_to_preprocessed_files: Path,
    channel: str,
) -> pd.DatetimeIndex:
    """
    Return the set of dates for which preprocessed files already exist.

    Parameters
    ----------
    path_to_preprocessed_files : Path
        Directory containing preprocessed ``.npy`` files.
    channel : str
        Channel identifier (e.g. ``'aia_171'``, ``'hmi'``).

    Returns
    -------
    pd.DatetimeIndex
        Sorted index of dates with existing preprocessed files.
    """
    channel_str = channel.split('_')[1] if 'aia' in channel else channel

    preprocessed_files = sorted(os.listdir(path_to_preprocessed_files))
    existing_preprocessed_dates = []
    for file in preprocessed_files:
        if file.endswith('.npy') and file[:3] == channel_str:
            file_date, _, _ = read_file_name(file, preprocessed=True)
            existing_preprocessed_dates.append(file_date)

    return pd.DatetimeIndex(existing_preprocessed_dates)

def load_existing_raw_files(path_to_raw_files: Path) -> pd.Series:
    """
    Build an index of existing raw FITS files, keyed by observation date.

    Duplicate observation dates are deduplicated, keeping the first occurrence.

    Parameters
    ----------
    path_to_raw_files : Path
        Directory containing raw FITS files.

    Returns
    -------
    pd.Series
        Series with observation datetime as index and filename as values.
    """
    if not os.path.isdir(path_to_raw_files):
        raw_files = []
    else:
        raw_files = sorted(os.listdir(path_to_raw_files))

    raw_file_dates = []
    raw_file_names = []
    for file in raw_files:
        if file.endswith('.fits'):
            file_date, _, _ = read_file_name(file)
            raw_file_dates.append(file_date)
            raw_file_names.append(file)

    dates_index = pd.DatetimeIndex(raw_file_dates)
    # Keep only the first file for each observation time
    unique_mask = ~dates_index.duplicated(keep='first')
    return pd.Series(
        pd.Index(raw_file_names)[unique_mask],
        index=dates_index[unique_mask],
    )


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

        file_date, _, _ = read_file_name(file)
        fits_file = path_to_downloaded / file

        try:
            smap = sunpy.map.Map(fits_file)
            reading_success = True
        except Exception:
            reading_success = False

        if reading_success:
            full_disk = contains_full_disk(smap)
            good_quality = smap.meta['QUALITY'] == 0

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
        file_date = datetime.strptime(date_str, '%Y-%m-%d_%H:%M')

    else:
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

    return file_date, product, channel


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

        good_dates, _ = check_file_quality(list(files_to_check), path_to_raw_files)

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
        delayed(find_substitute_file)(date, existing_raw_files, path_to_raw_files)
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
    date_str = target_date.strftime('%Y-%m-%d_%H:%M')
    base_name = f'{channel}_{date_str}'

    np.save(path_output / f'{base_name}.npy', image)
    with open(path_output / f'{base_name}_meta.pickle', 'wb') as f:
        pickle.dump(metadata, f)

    print(f'Saved {base_name}.npy')
