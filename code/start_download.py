import jsoc_download as jd
from datetime import datetime,timedelta
from dateutil.relativedelta import relativedelta
import os
import numpy as np
import pandas as pd
import pickle
from copy import copy
import time
import skimage
import matplotlib.pyplot as plt
import sunpy.map
from sunpy.map import contains_full_disk
from utils import read_file_name


def check_file_quality(files, path_to_downloaded):
    existing_dates = []
    bad_dates = []

    for file in files:
        if file[-5:] == '.fits':
            file_date, product, channel = read_file_name(file)

            fits_file = path_to_downloaded + file

            # check if images can be read and preprocessed
            try:
                #print('Reading', fits_file)
                aia_map = sunpy.map.Map(fits_file)
                reading_success = True
            except:
                print('Fits file could not be read.', product, channel, file_date)
                reading_success = False

            if reading_success:
                full_disk = True
                if not contains_full_disk(aia_map):
                    print('Map does not contain full disk.', product, channel, file_date)
                    full_disk = False

                bad_quality = False
                if not aia_map.meta['QUALITY'] == 0:
                    print('Map has bad quality.', product, channel, file_date)
                    bad_quality = True

                if full_disk and not bad_quality:
                    existing_dates.append(file_date)
                else:
                    bad_dates.append(file_date)
            else:
                bad_dates.append(file_date)
    return existing_dates, bad_dates


def download_adjacent_images(date, path_downloaded_month, interval, client, series, wavelength, segment):
    start_date = date - timedelta(minutes=interval)
    if series[:3] == 'AIA':
        end_date = date + timedelta(minutes=interval + 5)
    elif series[:3] == 'hmi':
        end_date = date + timedelta(minutes=interval+12)

    cadence = timedelta(minutes=interval * 2)
    # download data
    start_time = time.time()
    client = jd.client(client)
    try:
        rs = client.create_request_string(series,
                                          start_date,
                                          endtime=end_date,
                                          wavelength=wavelength,
                                          segment=segment,
                                          period='',
                                          cadence=cadence)
        search_results = client.search(rs, keys=['t_obs', '**ALL**'])
        print(search_results)
    except:
        print('Failed to request',start_date,'to',end_date,'with',cadence,'cadence.')
        return []
    
    if search_results.empty:
        return []

    pickle.dump(search_results,
                open(path_downloaded_month + '/meta_data_{}.pickle'.format(start_date.strftime('%Y%m%d%H%M')), 'wb'))
    try:
        files_downloaded = client.download(rs, path_downloaded_month,
                                           method='url-tar',
                                           protocol='fits',
                                           filter=None,
                                           rebin=rebin,
                                           process={})

        print(files_downloaded)
        print("--- %s seconds ---" % (time.time() - start_time))
    except:
        print('Failed to download',start_date,'to',end_date,'with',cadence,'min cadence.')
        return []

    return files_downloaded


def check_for_missing_images(month,path_to_downloaded,path_to_preprocessed,client,series, wavelength, segment):

    path_downloaded_month = path_to_downloaded + month.strftime('%Y/%m')
    if month.year == 2010 and month.month == 5:
        month_start = datetime(month.year, month.month, 13, 0)
        month_end = datetime(2010, 5, 31, 23)
    else:
        month_start = datetime(month.year, month.month, 1, 0)
        month_end = month_start+relativedelta(months=1)-timedelta(hours=1)


    target_data = pd.date_range(month_start,month_end,freq='1h')
    files = os.listdir(path_downloaded_month)
    files.sort()
    existing_hourly_dates = []

    existing_small_cadence_files = []
    existing_small_cadence_dates = []

    for file in files:
        if file[-5:] == '.fits':
            file_date, _, _ = read_file_name(file)
            existing_hourly_dates.append(pd.Timestamp(file_date).round('60min').to_pydatetime())
            existing_small_cadence_files.append(file)
            if series[:3] == 'AIA':
                existing_small_cadence_dates.append(pd.Timestamp(file_date).round('5min').to_pydatetime())
            elif series[:3] == 'hmi':
                existing_small_cadence_dates.append(pd.Timestamp(file_date).round('12min').to_pydatetime())


    existing_downloaded_hourly_dates = pd.Index(existing_hourly_dates)
    existing_downloaded_small_cadence = pd.Series(existing_small_cadence_dates,index=existing_small_cadence_files)

    missing_downloaded_data = target_data.difference(existing_downloaded_hourly_dates)

    missing_preprocessed_dates = pickle.load(open(path_to_preprocessed + month.strftime('%Y/%m') + '/bad_quality_dates.pickle','rb'))
    missing_preprocessed_dates = pd.Index(missing_preprocessed_dates).round('60min').to_pydatetime()

    missing_dates = missing_downloaded_data.union(missing_preprocessed_dates)
    print(missing_dates)

    substitute_dates = pd.Series(np.zeros(len(missing_dates),dtype=datetime),index=missing_dates)
    for date in missing_dates:
        substitute_success = False
        downloaded_all_candidates = False
        # check if there are already images downloaded around the missing images that could be used as substitute data
        mask = (existing_downloaded_small_cadence.values > date - timedelta(minutes=49)) & (existing_downloaded_small_cadence.values < date + timedelta(minutes=49))
        if np.sum(mask) > 1:
            print('Missing image:',date,', substitute candidates found.')
            file_list_missing_date = existing_downloaded_small_cadence.index[mask]

            path_to_files = path_to_downloaded + month.strftime('%Y/%m') + '/'
            good_dates, bad_dates = check_file_quality(file_list_missing_date, path_to_files)
            # check if dates until plus and minus 45 min were already downloaded and tested
            existing_downloaded_small_cadence_dates = existing_downloaded_small_cadence.loc[mask].sort_values().values
            if series[:3] == 'AIA':
                if (date - timedelta(minutes=45)) in existing_downloaded_small_cadence_dates and (date + timedelta(minutes=45)) in existing_downloaded_small_cadence_dates:
                    downloaded_all_candidates = True
            elif series[:3] == 'hmi':
                if (date - timedelta(minutes=48)) in existing_downloaded_small_cadence_dates and (date + timedelta(minutes=48)) in existing_downloaded_small_cadence_dates:
                    downloaded_all_candidates = True
            if len(good_dates) > 0:
                substitute_success = True
        else:
            print('Missing image:',date,', no substitute candidates found.')


        if not substitute_success and not downloaded_all_candidates:
            print('Missing image:',date,', downloading adjacent images.')
            continue_downloading = True
            if series[:3] == 'AIA':
                interval = 5
            elif series[:3] == 'hmi':
                interval = 12
            while continue_downloading:
                print('Downloading images', interval, 'minutes around',date)
                file_list_missing_date = download_adjacent_images(date, path_downloaded_month, interval,client, series, wavelength, segment)
                path_to_files = path_to_downloaded + month.strftime('%Y/%m') + '/'
                good_dates, bad_dates = check_file_quality(file_list_missing_date, path_to_files)
                if series[:3] == 'AIA':
                    interval = interval + 5
                elif series[:3] == 'hmi':
                    interval = interval + 12
                print(len(good_dates),len(bad_dates))
                if interval > 48 or len(good_dates) > 0:
                    continue_downloading = False

        # if none of the downloaded files can be used, mark date as discarded time stamp
        if len(good_dates) == 0:
            substitute_dates.loc[date] = None
            print('No substitute date for',date,'found.')
        else:
            time_dif = np.abs(date - pd.Index(good_dates))
            best_substitute = good_dates[np.argmin(time_dif)]
            substitute_dates.loc[date] = best_substitute
            print('Substitute date for',date,'succesfully determined:', best_substitute)

    #print(substitute_dates)
    substitute_dates.to_csv(path_downloaded_month+ '/substitute_dates.csv')

    return

def collect_substitute_dates(path_to_downloaded):
    start_month = datetime(2010,5,1)
    end_month = datetime(2024,7,31)

    current_month = copy(start_month)
    collect_subdat = []
    while current_month <= end_month:
        #print('Collecting',current_month.strftime('%Y/%m') + '/substitute_dates.csv')
        try:
            subdat_month = pd.read_csv(path_to_downloaded + current_month.strftime('%Y/%m') + '/substitute_dates.csv', index_col=0, parse_dates=[0, 1]).squeeze()
            collect_subdat.append(subdat_month)
        except:
            print(current_month.strftime('%Y/%m'), 'does not exist!')
        current_month = current_month + relativedelta(months=1)
    substitute_dates = pd.concat(collect_subdat,axis=0)
    substitute_dates.to_csv(path_to_downloaded + 'substitute_dates_merged.csv')
    print(substitute_dates)
    return substitute_dates



dir_path = os.path.dirname(os.getcwd())


start_date = datetime(2016,3,1)
run_on_server = True
channel = 'hmi'

if channel == 'aia_193':
    # last start from 2018
    client = 'daniel.collin@web.de'#'d.collin@tu-berlin.de' mp2 2018 #'collin@gfz-potsdam.de' mp3 2014 #'collin@gfz.de' mp3 2016 #'daniel.collin@web.de' mp2 2022/06
    series = 'AIA.lev1_euv_12s' #'hmi.M_720s'
    segment =  'image' # 'magnetogram'
    wavelength = 193 # ''
elif channel == 'aia_211':
    # last start from 2023
    client = 'daniel.collin@web.de'#'collin@gfz.de'
    series = 'AIA.lev1_euv_12s' #'hmi.M_720s'
    segment = 'image' # 'magnetogram'
    wavelength = 211 # ''
elif channel == 'hmi':
    client = 'collin@gfz.de' #'d.collin@tu-berlin.de' mp3 2017 # 'daniel.collin@gfz.de' mp1 2010 # collin@gfz.de mp3 2016/3 # 'collin@gfz-potsdam.de' mp3 2019/6 # 'daniel.collin@web.de' mp3 2022
    series = 'hmi.M_720s'  # 'AIA.lev1_euv_12s' #'hmi.M_720s'
    segment = 'magnetogram'  # 'image' # 'magnetogram'
    wavelength = ''  # 193 # ''

rebin = 4 # for 1024x1024 images # 16  for 256x256 images
if run_on_server:
    if series[0:3] == 'AIA':
        path_to_downloaded =  dir_path + '/data/SDO/AIA/{}/'.format(wavelength) #'SDO/HMI/magnetogram/'
        path_to_preprocessed = dir_path + '/processed_data/deep_learning/aia_{}/'.format(wavelength)

    elif series[0:3] == 'hmi':
        path_to_downloaded =  dir_path + '/data/SDO/HMI/magnetogram/' #'SDO/HMI/magnetogram/'
        path_to_preprocessed = dir_path + '/processed_data/deep_learning/hmi/'

else:
    if series[0:3] == 'AIA':
        path_to_downloaded =  dir_path + '/data/raw_data/aia_{}/'.format(wavelength) #'SDO/HMI/magnetogram/'
        path_to_preprocessed =  dir_path + '/data/processed_data/deep_learning/aia_{}/'.format(wavelength) #'SDO/HMI/magnetogram/'
    elif series[0:3] == 'hmi':
        path_to_downloaded =  dir_path + '/data/raw_data/hmi/'
        path_to_preprocessed =  dir_path + '/data/processed_data/deep_learning/hmi/'
if not os.path.isdir(path_to_downloaded):
    os.mkdir(path_to_downloaded)
if not os.path.isdir(path_to_preprocessed):
    os.mkdir(path_to_preprocessed)

#download_adjacent_images(datetime(2015,6,1), path_to_downloaded, 48, client, series, wavelength, segment)
#exit()
#collect_substitute_dates(path_to_downloaded)
#exit()
while start_date < datetime.today():
    month = datetime(start_date.year,start_date.month,1,0)
    check_for_missing_images(month, path_to_downloaded, path_to_preprocessed,client,series, wavelength, segment)
    start_date = start_date + relativedelta(months=1)
exit()


while start_date < datetime.today():
    start_time = time.time()

    end_date = start_date + timedelta(hours=24) #relativedelta(months=1)
    year = str(start_date.year)
    month = str(start_date.month)
    if len(month) == 1:
        month = '0' + month

    path_year = path_to_downloaded + year
    if not os.path.isdir(path_year):
        os.mkdir(path_year)
    path_month = path_year + '/' + month
    if not os.path.isdir(path_month):
        os.mkdir(path_month)
    path_month = path_month + '/'


    client = jd.client(client)
    rs = client.create_request_string(series,
                                      start_date,
                                      endtime=end_date,
                                      wavelength=wavelength,
                                      segment=segment,
                                      period='',
                                      cadence=timedelta(hours=1))
    search_results = client.search(rs, keys = ['t_obs','**ALL**'])
    print(search_results)
    pickle.dump(search_results,open(path_month + 'meta_data_{}.pickle'.format(start_date.strftime('%Y%m%d')),'wb'))

    files_downloaded = client.download(rs,path_month,
                                       method='url-tar',
                                       protocol='fits',
                                       filter=None,
                                       rebin=rebin,
                                       process={})

    print(files_downloaded)
    start_date = copy(end_date)
    print("--- %s seconds ---" % (time.time() - start_time))




