from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import sunpy.map
from sunpy.map import contains_full_disk
import os
import pandas as pd
import numpy as np
import pickle
import aiapy
from aiapy.calibrate import register, update_pointing, correct_degradation
from aiapy.calibrate.util import get_correction_table
from astropy.time import Time
import astropy.units as u
from rebin_psf import rebin_psf
from deconvolve_image import deconvolve_bid
from joblib import Parallel, delayed, cpu_count
from copy import copy

def create_folders_for_preprocessed_images(start,end,path_to_preprocessed):
    # create folders
    if not os.path.isdir(path_to_preprocessed):
        os.mkdir(path_to_preprocessed)

    current_month = copy(start)

    while current_month < end:
        if not os.path.isdir(path_to_preprocessed + current_month.strftime('%Y')):
            os.mkdir(path_to_preprocessed + current_month.strftime('%Y'))

        if not os.path.isdir(path_to_preprocessed + current_month.strftime('%Y/%m')):
            os.mkdir(path_to_preprocessed + current_month.strftime('%Y/%m'))
            #print('Creating',path_to_preprocessed + current_month.strftime('%Y/%m'))

        current_month = current_month + relativedelta(months=1)
    return



def find_missing_cropped_dates(month, path_to_cropped_files, channel):
    if 'aia' in channel:
        channel_str = channel.split('_')[1]
    else:
        channel_str = channel

    if month.year == 2010 and month.month == 5:
        month_start = datetime(2010, 5, 18)
        month_end = datetime(2010, 5, 31, 23)
    else:
        month_start = datetime(month.year, month.month, 1, 0)
        month_end = month_start + relativedelta(months=1) - timedelta(hours=1)
    target_data = pd.date_range(month_start, month_end, freq='1h')

    cropped_files = os.listdir(path_to_cropped_files)
    cropped_files.sort()
    existing_cropped_dates = []
    for file in cropped_files:
        if file[-4:] == '.npy' and file[:3] == str(channel_str):
            file_date, _, _ = read_file_name(file, preprocessed=True)
            existing_cropped_dates.append(file_date)

    existing_cropped_dates = pd.Index(existing_cropped_dates)

    missing_cropped_dates = target_data.difference(existing_cropped_dates)
    return missing_cropped_dates, existing_cropped_dates, target_data


def find_missing_preprocessed_dates(month, path_to_preprocessed_files, channel,overwrite_existing=False):
    if 'aia' in channel:
        channel_str = channel.split('_')[1]
    else:
        channel_str = channel

    if month.year == 2010 and month.month == 5:
        month_start = datetime(2010, 5, 18)
        month_end = datetime(2010, 5, 31, 23)
    else:
        month_start = datetime(month.year, month.month, 1, 0)
        month_end = month_start + relativedelta(months=1) - timedelta(hours=1)
    target_data = pd.date_range(month_start, month_end, freq='1h')

    if overwrite_existing:
        return target_data, target_data

    preprocessed_files = os.listdir(path_to_preprocessed_files)
    preprocessed_files.sort()
    existing_preprocessed_dates = []
    for file in preprocessed_files:
        if file[-4:] == '.npy' and file[:3] == str(channel_str):
            file_date, _, _ = read_file_name(file, preprocessed=True)
            existing_preprocessed_dates.append(file_date)

    existing_preprocessed_dates = pd.Index(existing_preprocessed_dates)

    missing_preprocessed_dates = target_data.difference(existing_preprocessed_dates)
    return missing_preprocessed_dates, target_data

def load_existing_preprocessed_dates(path_to_preprocessed_files, channel):
    if 'aia' in channel:
        channel_str = channel.split('_')[1]
    else:
        channel_str = channel

    preprocessed_files = os.listdir(path_to_preprocessed_files)
    preprocessed_files.sort()
    existing_preprocessed_dates = []
    for file in preprocessed_files:
        if file[-4:] == '.npy' and file[:3] == str(channel_str):
            file_date, _, _ = read_file_name(file, preprocessed=True)
            existing_preprocessed_dates.append(file_date)

    existing_preprocessed_dates = pd.Index(existing_preprocessed_dates)

    return existing_preprocessed_dates

def load_existing_raw_files(path_to_raw_files):
    if not os.path.isdir(path_to_raw_files):
        raw_files = []
    else:
        raw_files = os.listdir(path_to_raw_files)
    raw_files.sort()
    raw_file_dates = []
    # raw_file_dates_rounded = []
    raw_file_names = []
    for file in raw_files:
        if file[-5:] == '.fits':
            file_date, _, _ = read_file_name(file)
            raw_file_dates.append(file_date)
            raw_file_names.append(file)


    mask_unique_dates= np.invert(pd.Index(raw_file_dates).duplicated(keep='first'))
    existing_raw_files = pd.Series(pd.Index(raw_file_names)[mask_unique_dates], index=pd.Index(raw_file_dates)[mask_unique_dates])
    return existing_raw_files

def load_config_data(path_to_config,wl,month=None):
    print('Loading PSF.')

    # try loading rebinned PSF
    rebin_dimension = [1024, 1024]
    try:
        psf_rebinned = pickle.load(
            open(path_to_config / 'psf_{}_{}x{}.pickle'.format(wl, rebin_dimension[0], rebin_dimension[1]), 'rb'))
        # print('Found saved rebinned PSF.')
    except:
        # if not found, try loading regular full scale psf
        try:
            psf = pickle.load(open(path_to_config / 'psf_{}.pickle'.format(wl), 'rb'))
            # print('Found saved PSF.')
        except:
            use_gpu = True
            print('Did not find saved PSF. Computing it from scratch. This may take a while. GPU usage is strongly recommended.')
            if use_gpu:
                print('GPU usage is currently enabled.')
                print('If this generates an error or there is no GPU support available, consider disabling GPU usage in the script or computing the PSF on a different device with GPU support and copying it into the configuration_data folder. ')
            else:
                print('GPU usage is currently disabled. This might be very slow. Consider enabling GPU usage in the script.')

            psf = aiapy.psf.psf(int(wl) * u.angstrom, use_gpu=use_gpu)
            pickle.dump(psf, open(path_to_config / 'psf_{}.pickle'.format(wl), 'wb'))

        # print('Not found saved rebinned PSF. Compute it from from original PSF.')
        psf_rebinned = rebin_psf(psf, rebin_dimension)
        pickle.dump(psf_rebinned,
                    open(path_to_config / 'psf_{}_{}x{}.pickle'.format(wl, rebin_dimension[0], rebin_dimension[1]),
                         'wb'))

    print('Loading degradation correction table.')
    try:
        correction_table = pickle.load(open(path_to_config / 'degradation_correction_table.pickle', 'rb'))
        #print('Found saved degradation table.')
    except:
        print('Did not find saved degradation table. Downloading it.')
        correction_table = get_correction_table("JSOC")
        pickle.dump(correction_table, open(path_to_config / 'degradation_correction_table.pickle', 'wb'))

    if month is not None:
        print('Loading pointing table for {}.'.format(month.strftime('%Y%m')))
        try:
            pointing_table = pickle.load(
                open(path_to_config / 'pointing_table_{}.pickle'.format(month.strftime('%Y%m')), 'rb'))
            #print('Found saved pointing table.')
        except:
            print('Did not find saved pointing table. Downloading it.')
            # print(Time(current_month-timedelta(days=1)),Time(current_month+timedelta(days=32)))
            time_range = (Time(month - timedelta(days=1)), Time(month + timedelta(days=32)))
            pointing_table = aiapy.calibrate.util.get_pointing_table("JSOC", time_range=time_range)
            pickle.dump(pointing_table,
                        open(path_to_config / 'pointing_table_{}.pickle'.format(month.strftime('%Y%m')), 'wb'))
    else:
        pointing_table = None

    return psf_rebinned, correction_table, pointing_table


def check_file_quality(files, path_to_downloaded):
    existing_dates = []
    bad_dates = []

    for file in files:
        if file[-5:] == '.fits':
            file_date, product, channel = read_file_name(file)

            fits_file = path_to_downloaded / file

            # check if images can be read and preprocessed
            try:
                #print('Reading', fits_file)
                aia_map = sunpy.map.Map(fits_file)
                reading_success = True
            except:
                #print('Fits file could not be read.', product, channel, file_date)
                reading_success = False

            if reading_success:
                full_disk = True
                if not contains_full_disk(aia_map):
                    #print('Map does not contain full disk.', product, channel, file_date)
                    full_disk = False

                bad_quality = False
                if not aia_map.meta['QUALITY'] == 0:
                    #print('Map has bad quality.', product, channel, file_date)
                    bad_quality = True

                if full_disk and not bad_quality:
                    existing_dates.append(file_date)
                    return existing_dates, bad_dates
                else:
                    bad_dates.append(file_date)
            else:
                bad_dates.append(file_date)
    return existing_dates, bad_dates

def read_file_name(file,preprocessed=False):
    file = str(file)
    if preprocessed:
        if file[0:3] == '171' or file[0:3] == '193' or file[0:3] == '211':
            product = 'aia'
            channel = file[0:3]
        elif file[0:3] == 'hmi':
            product = 'hmi'
            channel = ''
        if '.pickle' in file:
            date_str = file.split('.pickle')[0][4:]
        elif '.npy' in file:
            date_str = file.split('.npy')[0][4:]
        file_date = datetime.strptime(date_str, '%Y-%m-%d_%H:%M')
    else:
        product = file[0:3]
        if product == 'hmi':
            channel = ''
            date_str = file.split('720s.')[1][:8]
            time_str = file.split('_TAI')[0][-6:]
            if int(time_str[-2:]) > 59:
                time_list = list(time_str)
                time_list[-2:] = '59'
                time_str = ''.join(time_list)
            file_date = datetime.strptime(date_str + '_' + time_str, '%Y%m%d_%H%M%S')
        elif product == 'aia':
            try:
                product = file[0:3]
                channel = file.split('.image')[0][-3:]
                date_str = file.split('T')[0][-10:]
                time_str = file.split('T')[1][:6]

                if int(time_str[-2:]) > 59:
                    time_list = list(time_str)
                    time_list[-2:] = '59'
                    time_str = ''.join(time_list)
                file_date = datetime.strptime(date_str + '_' + time_str, '%Y-%m-%d_%H%M%S')
            except:
                product = file[0:3]
                channel = file.split('lev1_')[1][:3]
                date_str = file.split('t')[0][-10:]
                time_str = file.split('t')[1][:8]
                if int(time_str[-2:]) > 59:
                    time_list = list(time_str)
                    time_list[-2:] = '59'
                    time_str = ''.join(time_list)
                file_date = datetime.strptime(date_str + '_' + time_str, '%Y_%m_%d_%H_%M_%S')
    return file_date, product, channel


def check_completeness_of_preprocessed_images(files_to_exclude,current_month, path_to_preprocessed_files, channel):
    # load again missing dates
    missing_preprocessed_dates, target_data = find_missing_preprocessed_dates(current_month, path_to_preprocessed_files,
                                                                              channel, overwrite_existing=False)

    dates_to_check = []
    # check if each missing date is either in missing raw maps or bad raw candidates
    for missing_date in missing_preprocessed_dates:
        # maybe create overview of target dates, with indicator if it exists, bad, missing raw, and meta data etc.
        check_date = True
        if missing_date in files_to_exclude.index:
            if files_to_exclude.loc[missing_date, 'bad'] or files_to_exclude.loc[missing_date, 'missing_raw']:
                check_date = False
        if check_date:
            dates_to_check.append(missing_date)

    if len(dates_to_check) == 0:
        print('All possible dates for {} successfully preprocessed.'.format(current_month.strftime('%Y/%m')))
        all_successfully_preprocessed = True
    else:
        print('The following dates are downloaded and of good quality but have not been preprocessed successfully:')
        print(dates_to_check)
        all_successfully_preprocessed = False
    return all_successfully_preprocessed, dates_to_check

def parallel_find_files_to_preprocess(missing_date,existing_raw_files, path_to_raw_files):

    print('Checking',missing_date)

    bad_raw_candidates = False
    missing_raw_map = False
    #files_to_preprocess_date = None
    files_to_preprocess_name = None

    mask = (existing_raw_files.index > (missing_date - timedelta(hours=24,minutes=30))) & (
                existing_raw_files.index < (missing_date + timedelta(hours=24,minutes=30)))
    if np.sum(mask) > 0:
        #print('Candidate maps existing for:', missing_date)
        #print(existing_raw_files.loc[mask])
        # print('Checking file quality.')
        files_to_check = existing_raw_files.loc[mask]
        # sort by temporal difference to missing date
        time_difference = np.abs(missing_date - files_to_check.index)
        sort_index = np.argsort(time_difference)
        files_to_check = files_to_check.iloc[sort_index]
        good_dates, bad_dates = check_file_quality(list(files_to_check), path_to_raw_files)

        #print('Good candidates:', len(good_dates), 'Bad candidates:', len(bad_dates))
        if len(good_dates) > 0:
            time_differences = np.abs(missing_date - pd.Index(good_dates))
            #print('Time differences to good candidate maps:')
            #print(time_differences)
            # Only preprocess the temporally closest good date!
            best_candidate_date = good_dates[np.argmin(time_differences)]
            #print('Best substitute candidate:', best_candidate_date, 'Time difference:', np.min(time_differences))

            file_name = existing_raw_files.loc[best_candidate_date]
            #print('Preprocessing:', best_candidate_date, 'File:', file_name)
            #files_to_preprocess_date.append(missing_date)
            #files_to_preprocess_name.append(file_name)
            #files_to_preprocess_date = missing_date
            files_to_preprocess_name = file_name
        else:
            #print('No good candidates found for', missing_date)
            #bad_raw_candidates.append(missing_date)
            bad_raw_candidates = True
    else:
        #print('No raw maps found for', missing_date)
        #missing_raw_maps.append(missing_date)
        missing_raw_map = True

    return missing_date, files_to_preprocess_name, bad_raw_candidates, missing_raw_map

def find_files_to_preprocess(missing_preprocessed_dates, existing_raw_files, path_to_raw_files):
    files_to_preprocess_name = []
    files_to_preprocess_date = []
    missing_raw_maps = []
    bad_raw_candidates = []

    n_cpus = cpu_count()
    print('Analyzing files to find files to preprocess.')
    print('Number of available CPUs:', n_cpus)

    res = Parallel(n_jobs=int(n_cpus / 2))(
        delayed(parallel_find_files_to_preprocess)(date, existing_raw_files, path_to_raw_files) for date in missing_preprocessed_dates)

    res = np.array(res)
    res = pd.DataFrame(res[:,1:],index=res[:,0],columns=['file_name','bad','missing_raw'])#,dtype=[str,bool,bool])

    print('Files to preprocess:')
    files_to_preprocess = res.loc[:,'file_name']#.dropna()
    nan_mask = pd.isna(files_to_preprocess).values
    files_to_preprocess = pd.Series(files_to_preprocess.index[np.invert(nan_mask)],index=files_to_preprocess.values[np.invert(nan_mask)])
    print(files_to_preprocess)

    print('Dates that cannot be preprocessed:')
    files_to_exclude = res.loc[nan_mask,['bad','missing_raw']]
    print(files_to_exclude)

    return files_to_preprocess, files_to_exclude