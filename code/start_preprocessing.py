from datetime import datetime,timedelta
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
import os
import numpy as np
import pandas as pd
import pickle
from rebin_psf import rebin_psf
from deconvolve_image import deconvolve_bid
import aiapy.psf
from aiapy.calibrate import register, update_pointing, correct_degradation
from aiapy.calibrate.util import get_correction_table
import sunpy.map
from sunpy.map import contains_full_disk
import astropy.units as u
import time
from joblib import Parallel, delayed, cpu_count
from skimage.transform import rescale, resize
from skimage.measure import block_reduce
from matplotlib.patches import Circle
import cv2
from copy import copy
from utils import (read_file_name, check_file_quality, find_missing_preprocessed_dates, load_existing_raw_files,
                   load_config_data, find_files_to_preprocess, check_completeness_of_preprocessed_images,
                   find_missing_cropped_dates, load_existing_preprocessed_dates,
                   create_folders_for_preprocessed_images)
from astropy.time import Time
from pathlib import Path
from sunpy.coordinates import Helioprojective, SphericalScreen, propagate_with_solar_surface
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from sunpy.physics.differential_rotation import solar_rotate_coordinate



def register_image(smap, missing=None,arcsec_pix_target=None):
    """
    Register an SDO image.
    """

    orig_shape = smap.data.shape[0]

    # Default pixel scale: 0.6 arcsec/pixel scaled by resolution
    if arcsec_pix_target is None:
        downsample_factor = 4096 / orig_shape
        arcsec_pix_target = 0.6 * downsample_factor

    if arcsec_pix_target is None and np.log2(smap.data.shape[0]) % 1 > 0:
        print('Warning: Map shape is not a power of 2. Please specify a target arcsec per pixel size.')

    scale = arcsec_pix_target * u.arcsec
    scale_factor = smap.scale[0] / scale

    missing = smap.min() if missing is None else missing
    tempmap = smap.rotate(
        recenter=True,
        scale=scale_factor.value,
        order=3,
        missing=missing,
        method='scipy',
    )
    # extract center from padded smap.rotate output
    # crpix1 and crpix2 will be equal (recenter=True)
    center = np.floor(tempmap.meta["crpix1"])
    range_side = (center + np.array([-1, 1]) * smap.data.shape[0] / 2) * u.pix
    newmap = tempmap.submap(
        u.Quantity([range_side[0], range_side[0]]),
        top_right=u.Quantity([range_side[1], range_side[1]]) - 1 * u.pix,
    )
    newmap.meta["r_sun"] = newmap.meta["rsun_obs"] / newmap.meta["cdelt1"]
    newmap.meta["lvl_num"] = 1.5
    newmap.meta["bitpix"] = -64

    if newmap.data.shape[0] > orig_shape:
        big_map = np.array(newmap.data)
        center_pixel = big_map.shape[0] / 2 - 0.5
        cutout_range = np.array([center_pixel - orig_shape / 2, center_pixel + orig_shape / 2])
        cutout_range[0] = cutout_range[0] + 0.5
        cutout_range[1] = cutout_range[1] + 0.5
        cutout_range = cutout_range.astype(int)
        big_map_cut = big_map[cutout_range[0]:cutout_range[1], cutout_range[0]:cutout_range[1]]
        newmap = sunpy.map.Map(big_map_cut,newmap.meta)
    elif newmap.data.shape[0] < orig_shape:
        small_map = np.array(newmap.data)
        pad_width = int((orig_shape - small_map.shape[0]) / 2)
        # Add padding
        small_map_padded = np.pad(small_map, pad_width, mode='constant', constant_values=0.)
        newmap = sunpy.map.Map(small_map_padded, newmap.meta)

    newmap.meta["crpix1"] = orig_shape / 2 + 0.5
    newmap.meta["crpix2"] = orig_shape / 2 + 0.5
    return newmap



def scale_solar_disk_radius(map, rsun_target=976.0, missing=None):
    # scales a registered map to a fixed given solar radius rsun_target (in arcsec per pixels). That simultaneously fixes the pixel radius of the solar disk.
    orig_shape = map.data.shape[0]

    scale_factor = rsun_target / map.meta['RSUN_OBS']
    missing = map.data.min() if missing is None else missing

    temp_map = map.rotate(scale=scale_factor, order=3, missing=missing, method='scipy')


    if temp_map.data.shape[0] > orig_shape:
        center_pixel = temp_map.data.shape[0] / 2 - 0.5
        cutout_range = np.array([center_pixel - orig_shape / 2, center_pixel + orig_shape / 2])
        cutout_range[0] = cutout_range[0] + 0.5
        cutout_range[1] = cutout_range[1] + 0.5
        cutout_range = cutout_range.astype(int)
        new_img_data = temp_map.data[cutout_range[0]:cutout_range[1], cutout_range[0]:cutout_range[1]]
    elif temp_map.data.shape[0] < orig_shape:
        pad_width = int((orig_shape - temp_map.data.shape[0]) / 2)
        # Add padding
        new_img_data = np.pad(temp_map.data, pad_width, mode='constant', constant_values=0.)
    elif temp_map.data.shape[0] == orig_shape:
        new_img_data = temp_map.data

    new_map = sunpy.map.Map(new_img_data, temp_map.meta)
    return new_map



def compute_differential_rotation(smap, target_date):
    out_frame = Helioprojective(observer=smap.observer_coordinate, obstime=Time(target_date),
                                rsun=smap.coordinate_frame.rsun)

    out_center = SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=out_frame)
    header = sunpy.map.make_fitswcs_header(smap.data.shape,
                                                    out_center,
                                                    reference_pixel=u.Quantity(smap.reference_pixel),
                                                    scale=u.Quantity(smap.scale),
                                                    rotation_matrix=smap.rotation_matrix,
                                                    instrument=smap.instrument,
                                                    exposure=smap.exposure_time)
    out_wcs = WCS(header)

    with propagate_with_solar_surface():
        smap_reprojected = smap.reproject_to(out_wcs)

    new_map = sunpy.map.Map(smap_reprojected.data, smap.meta)

    return new_map

class AIAPreprocessor:
    def __init__(self,pointing_table,point_spread_function,correction_table):
        self.pointing_table = pointing_table
        self.point_spread_function = point_spread_function
        self.correction_table = correction_table

    def preprocess_image(self,aia_map,map_date,target_date):
        # upsample to original dimension to perform pointing update and fix observer location (can only be done for full size images)
        orig_pixel_radius = aia_map.rsun_obs/aia_map.scale[0]
        new_dimensions = [4096, 4096] * u.pixel
        #print('Updating pointing.')
        aia_resampled_map = aia_map.resample(new_dimensions)

        aia_map_updated_pointing = update_pointing(aia_resampled_map, pointing_table=self.pointing_table)
        # aia_map_observer_fixed = fix_observer_location(aia_map_updated_pointing)

        # do differential rotation here to match the target time stamp if the image at the target time is not available
        if np.abs(map_date - target_date) > timedelta(minutes=6):
            aia_map_differentially_rotated = compute_differential_rotation(aia_map_updated_pointing,target_date)
        else:
            aia_map_differentially_rotated = aia_map_updated_pointing

        # then downsample map again to speed up deconvolution
        new_dimensions = [1024, 1024] * u.pixel
        # aia_map_observer_fixed_downsampled = aia_map_observer_fixed.resample(new_dimensions)
        aia_map_downsampled = aia_map_differentially_rotated.resample(new_dimensions)

        use_gpu = False
        '''
        if use_gpu:
            print('Deconvolving. GPU usage enabled. If there is no GPU device available, this may generate an error and GPU usage should be disabled in the script.')
        else:
            print('Deconvolving. GPU usage disabled. If a GPU is available, consider enabling GPU usage in the script for a significant speed-up.')
        '''
        # perform deconvolution and replace negative values in the corners of the image
        aia_map_deconvolved = deconvolve_bid(aia_map_downsampled.data, self.point_spread_function, use_gpu=use_gpu)
        aia_map_deconvolved[np.where(aia_map_deconvolved < 0)] = 0.0
        aia_map_deconvolved = sunpy.map.Map(aia_map_deconvolved, aia_map_downsampled.meta)

        #print('Registering')
        # register map, i.e., rotate, scale and translate to align all images
        aia_map_registered = register_image(aia_map_deconvolved)
        reg_pixel_radius = aia_map_registered.rsun_obs/aia_map_registered.scale[0]

        aia_map_scaled = scale_solar_disk_radius(aia_map_registered,rsun_target=976.0)
        scaled_pixel_radius = aia_map_scaled.rsun_obs/aia_map_scaled.scale[0]

        print(orig_pixel_radius,reg_pixel_radius,scaled_pixel_radius)

        #rsun = aia_map_registered.rsun_obs.value
        #wcs = aia_map_registered.wcs

        #print('Correcting for degradation.')
        # correct for instrument degradation
        aia_map_degradation_corrected = correct_degradation(aia_map_scaled, correction_table=self.correction_table)

        # normalize the image exposure time such that the units of the image are DN / pixel / s
        img_exposure_time_corrected = aia_map_degradation_corrected.data / aia_map_degradation_corrected.exposure_time
        #aia_map_exposure_time_corrected = sunpy.map.Map(aia_map_exposure_time_corrected, aia_map_degradation_corrected.meta)

        # data is provided upside-down, so it needs to be flipped such that the y-axis is aligned to solar north
        img_final = np.flipud(img_exposure_time_corrected.value)

        return img_final, aia_map_degradation_corrected.meta


class HMIPreprocessor:
    def __init__(self):
        pass

    def preprocess_image(self,hmi_map,map_date,target_date):
        orig_pixel_radius = hmi_map.rsun_obs / hmi_map.scale[0]
        new_dimensions = [4096, 4096] * u.pixel
        # print('Updating pointing.')
        hmi_resampled_map = hmi_map.resample(new_dimensions)

        # do differential rotation here to match the target time stamp if the image at the target time is not available
        if np.abs(map_date - target_date) > timedelta(minutes=6):
            hmi_map_differentially_rotated = compute_differential_rotation(hmi_resampled_map, target_date)
        else:
            hmi_map_differentially_rotated = hmi_resampled_map

        # then downsample map again
        new_dimensions = [1024, 1024] * u.pixel
        hmi_map_downsampled = hmi_map_differentially_rotated.resample(new_dimensions)

        hmi_map_registered = register_image(hmi_map_downsampled,missing=0.0)

        map_aligned = np.array(hmi_map_registered.data)
        map_aligned_without_nan = np.nan_to_num(map_aligned)
        hmi_map_aligned_without_nan = sunpy.map.Map(map_aligned_without_nan, hmi_map_registered.meta)

        reg_pixel_radius = hmi_map_aligned_without_nan.rsun_obs / hmi_map_aligned_without_nan.scale[0]

        hmi_map_scaled = scale_solar_disk_radius(hmi_map_aligned_without_nan, rsun_target=976.0, missing=0.0)
        scaled_pixel_radius = hmi_map_scaled.rsun_obs / hmi_map_scaled.scale[0]

        print(orig_pixel_radius, reg_pixel_radius, scaled_pixel_radius)

        # data is provided upside-down, so it needs to be flipped such that the y-axis is aligned to solar north
        img_final = np.flipud(hmi_map_scaled.data)

        return img_final, hmi_map_scaled.meta


def parallel_aia_preprocessing(file,files_to_preprocess,path_to_input_files,path_to_output_month,pointing_table,psf_rebinned,correction_table):

    if not file[-5:] == '.fits':
        return
    else:
        fits_file = path_to_input_files / file
        date, product, channel = read_file_name(file,preprocessed=False)
        # sometimes there are multiple target dates for the same image. That happens when an image is differentially rotated to multiple time points to fill gaps.
        target_dates = files_to_preprocess.loc[[file]].to_list()

        try:
            aia_map = sunpy.map.Map(fits_file)
        except:
            print('Fits file could not be read.', product, channel, date)
            return

        if not contains_full_disk(aia_map):
            print('Map does not contain full disk.', product, channel, date)
            return

        if not aia_map.meta['QUALITY'] == 0:
            print('Bad quality', product, channel, date)
            return
        else:
            resolution = aia_map.data.shape[0]
            if resolution == 4096:
                aia_map = aia_map.resample([1024, 1024] * u.pixel)

            print('Processing', product, channel, date)
            aia_preprocessor = AIAPreprocessor(pointing_table, psf_rebinned,correction_table)
            for target_date in target_dates:
                preprocessed_image, meta_info = aia_preprocessor.preprocess_image(aia_map,date,target_date)
                new_file_name = channel + '_' + target_date.strftime('%Y-%m-%d_%H:%M') + '.npy'
                np.save(path_to_output_month / new_file_name, preprocessed_image)
                pickle.dump(meta_info,open(path_to_output_month / new_file_name.replace('.npy','_meta.pickle'),'wb'))
                print('Saving', new_file_name)

            return


def parallel_hmi_preprocessing(file,files_to_preprocess,path_to_input_files,path_to_output_month):

    if not file[-5:] == '.fits':
        return
    else:
        fits_file = path_to_input_files / file
        date, product, channel = read_file_name(file, preprocessed=False)
        # sometimes there are multiple target dates for the same image. That happens when an image is differentially rotated to multiple time points to fill gaps.
        target_dates = files_to_preprocess.loc[[file]].to_list()

        try:
            hmi_map = sunpy.map.Map(fits_file)
        except:
            print('Fits file could not be read.', product, channel, date)
            return

        if not contains_full_disk(hmi_map):
            print('Map does not contain full disk.', product, channel, date)
            return

        if not hmi_map.meta['QUALITY'] == 0:
            print('Bad quality', product, channel, date)
            return
        else:

            resolution = hmi_map.data.shape[0]
            if resolution == 4096:
                hmi_map = hmi_map.resample([1024, 1024] * u.pixel)

            print('Processing', product, 'magnetogram', date)
            hmi_preprocessor = HMIPreprocessor()
            for target_date in target_dates:
                preprocessed_image, meta_info = hmi_preprocessor.preprocess_image(hmi_map, date, target_date)
                new_file_name = 'hmi_' + target_date.strftime('%Y-%m-%d_%H:%M') + '.npy'
                np.save(path_to_output_month / new_file_name, preprocessed_image)
                pickle.dump(meta_info, open(path_to_output_month / new_file_name.replace('.npy', '_meta.pickle'), 'wb'))
                print('Saving', new_file_name)

            return



def parallel_image_cropping(file,path_to_input_files,path_to_output,downsample_resolution=512,crop_square_in_downsampled=300,resize_cropped=224):
    if file[-4:] == '.npy':
        print('Processing image',file)

        img = np.load(path_to_input_files / file)

        current_resolution = img.shape[0] # images are expected to be square

        downsample_factor = current_resolution // downsample_resolution

        if current_resolution % downsample_resolution > 0:
            raise ValueError('The current image resolution must be divisible by the target downsampled resolution. Currently, only skimage block_reduce is implemented. Pleas modifiy if needed.')

        # downsample by summing in blocks
        img = block_reduce(img, (downsample_factor, downsample_factor), np.sum)

        # crop image
        if type(crop_square_in_downsampled) == int:
            cut_pixels = int((img.shape[0]-crop_square_in_downsampled)/2)
            img = img[cut_pixels:-cut_pixels,cut_pixels:-cut_pixels]
        elif type(crop_square_in_downsampled) == str and crop_square_in_downsampled == 'disk':
            meta_data = pickle.load(open(path_to_input_files / file.replace('.npy', '_meta.pickle'), 'rb'))
            downsampled_scale = meta_data['cdelt1'] * downsample_factor
            sun_radius_in_pixels = meta_data['rsun_obs'] / downsampled_scale
            cut_pixels = int(np.round((img.shape[0] - sun_radius_in_pixels*2) / 2,2))
            img = img[cut_pixels:-cut_pixels,cut_pixels:-cut_pixels]


        # resize image to desired pixels
        if resize_cropped is not None:
            img = resize(img,(resize_cropped,resize_cropped),order=3,mode='constant',cval=0)

        img = img.astype('float32')
        np.save(path_to_output / file, img)

    return




class SolarImagePreprocessor:
    def __init__(self,start, end, channels, paths):
        self.paths = paths
        self.start = start#datetime(2015,1,1) #datetime(2010,5,1)
        self.end = end#datetime(2015,1,31) #datetime.today() #
        self.channels = channels#['aia_171','aia_193','aia_211','hmi']#,
        #self.channels = ['hmi']


        self._create_folders()

    def _create_folders(self):
        if not os.path.isdir(self.paths['preprocessed']):
            os.mkdir(self.paths['preprocessed'])

        for channel in self.channels:
            channel_path = self.paths['preprocessed'] / channel
            if not os.path.isdir(channel_path):
                os.mkdir(channel_path)

            current_month = copy(self.start)

            while current_month < self.end:
                if not os.path.isdir(channel_path / current_month.strftime('%Y')):
                    os.mkdir(channel_path / current_month.strftime('%Y'))

                if not os.path.isdir(channel_path / current_month.strftime('%Y') / current_month.strftime('%m')):
                    os.mkdir(channel_path / current_month.strftime('%Y') / current_month.strftime('%m'))
                    # print('Creating',path_to_preprocessed + current_month.strftime('%Y/%m'))

                current_month = current_month + relativedelta(months=1)
        return

    def run(self,load_preprocessing_fails=False,overwrite_existing=False):
        for channel in self.channels:
            if 'aia' in channel:
                path_to_raw_channel = self.paths['raw'] / 'AIA' / channel.split('_')[1]
            elif 'hmi' in channel:
                path_to_raw_channel = self.paths['raw'] / 'HMI' / 'magnetogram'

            path_to_preprocessed_channel = self.paths['preprocessed'] / channel

            current_month = datetime(self.start.year, self.start.month, 1)

            collect_dates_to_check = []
            while current_month <= self.end:
                path_to_raw_files = path_to_raw_channel / current_month.strftime('%Y/%m')
                existing_raw_files = load_existing_raw_files(path_to_raw_files)

                path_to_preprocessed_files = path_to_preprocessed_channel / current_month.strftime('%Y/%m')
                missing_preprocessed_dates, target_data = find_missing_preprocessed_dates(current_month,
                                                                                          path_to_preprocessed_files,
                                                                                          channel,overwrite_existing)

                if load_preprocessing_fails:
                    try:
                        files_to_exclude_saved = pd.read_csv(path_to_preprocessed_files + 'preprocessing_fails.csv',
                                                             index_col=0, parse_dates=[0])
                        missing_preprocessed_dates = missing_preprocessed_dates.difference(files_to_exclude_saved.index)
                    except FileNotFoundError:
                        files_to_exclude_saved = pd.DataFrame([], columns=['bad', 'missing_raw'])
                else:
                    files_to_exclude_saved = pd.DataFrame([], columns=['bad', 'missing_raw'])

                if len(missing_preprocessed_dates) > 0:
                    files_to_preprocess, files_to_exclude = find_files_to_preprocess(missing_preprocessed_dates,
                                                                                     existing_raw_files, path_to_raw_files)
                    if not files_to_exclude_saved.empty:
                        files_to_exclude = pd.concat([files_to_exclude_saved, files_to_exclude], axis=0)
                    files_to_exclude.to_csv(path_to_preprocessed_files / 'preprocessing_fails.csv')
                else:
                    files_to_preprocess = []
                    files_to_exclude = files_to_exclude_saved

                # do preprocessing here
                if len(files_to_preprocess) > 0:
                    self._preprocess_missing_images(files_to_preprocess, current_month, path_to_raw_files,
                                              path_to_preprocessed_files, channel)

                # check if preprocessing has worked for all target dates besides the ones for which there are only bad quality dates available, or no images could be downloaded
                all_successfully_preprocessed, dates_to_check = check_completeness_of_preprocessed_images(files_to_exclude,
                                                                                                          current_month,
                                                                                                          path_to_preprocessed_files,
                                                                                                          channel)
                collect_dates_to_check.extend(dates_to_check)

                current_month = current_month + relativedelta(months=1)
            print('All good?')
            print(len(collect_dates_to_check) == 0)
            print(collect_dates_to_check)
        return

    def _preprocess_missing_images(self,files_to_preprocess, current_month, path_to_raw_files, path_to_preprocessed_files,
                                  channel):
        n_cpus = cpu_count()
        print('Number of available CPUs:', n_cpus)
        print('Start preprocessing.')

        if channel == 'hmi':
            Parallel(n_jobs=int(n_cpus / 2))(
                delayed(parallel_hmi_preprocessing)(file, files_to_preprocess, path_to_raw_files,
                                                    path_to_preprocessed_files) for file in files_to_preprocess.index.unique())
        else:
            psf, correction_table, pointing_table = load_config_data(self.paths['config'], channel.split('_')[1], current_month)
            Parallel(n_jobs=int(n_cpus / 2))(
                delayed(parallel_aia_preprocessing)(file, files_to_preprocess, path_to_raw_files,
                                                    path_to_preprocessed_files, pointing_table, psf,
                                                    correction_table) for file in files_to_preprocess.index.unique())

        return

    def _crop_images(self):


        return


class ImageCropper:
    def __init__(self,start,end,channels,paths,downsample_resolution=512,crop_square_in_downsampled=300,resize_cropped=224):
        self.paths = paths
        self.start = start
        self.end = end
        self.channels = channels
        self.downsample_resolution = downsample_resolution
        self.crop_square_in_downsampled = crop_square_in_downsampled
        self.resize_cropped=resize_cropped
        self._create_folders()

    def _create_folders(self):

        if not os.path.isdir(self.paths['cropped']):
            os.mkdir(self.paths['cropped'])

        for channel in self.channels:
            channel_path = self.paths['cropped'] / channel
            if not os.path.isdir(channel_path):
                os.mkdir(channel_path)

            current_month = copy(self.start)

            while current_month < self.end:
                if not os.path.isdir(channel_path / current_month.strftime('%Y')):
                    os.mkdir(channel_path / current_month.strftime('%Y'))

                if not os.path.isdir(channel_path / current_month.strftime('%Y') / current_month.strftime('%m')):
                    os.mkdir(channel_path / current_month.strftime('%Y') / current_month.strftime('%m'))
                    # print('Creating',path_to_preprocessed + current_month.strftime('%Y/%m'))

                current_month = current_month + relativedelta(months=1)
        return

    def run(self):
        for channel in self.channels:
            path_to_preprocessed_channel = self.paths['preprocessed'] / channel
            path_to_cropped_channel = self.paths['cropped'] / channel
            current_month = datetime(self.start.year, self.start.month, 1)
            collect_dates_to_check = []
            while current_month <= self.end:
                path_to_preprocessed_files = path_to_preprocessed_channel / current_month.strftime('%Y/%m')
                existing_preprocessed_dates = load_existing_preprocessed_dates(path_to_preprocessed_files, channel)

                path_to_cropped_files = path_to_cropped_channel / current_month.strftime('%Y/%m')

                missing_cropped_dates, existing_cropped_dates, target_data = find_missing_cropped_dates(current_month,
                                                                                                        path_to_cropped_files,
                                                                                                        channel)

                dates_to_crop = existing_preprocessed_dates.difference(existing_cropped_dates)
                files_to_crop_names = []
                if 'aia' in channel:
                    channel_str = channel.split('_')[1]
                else:
                    channel_str = channel
                for date in dates_to_crop:
                    file_name = str(channel_str) + '_' + date.strftime('%Y-%m-%d_%H:%M') + '.npy'
                    files_to_crop_names.append(file_name)

                # do cropping here
                if len(dates_to_crop) > 0:
                    self._crop_images(channel,path_to_preprocessed_files,path_to_cropped_files,files_to_crop_names)

                missing_preprocessed_dates, target_data = find_missing_preprocessed_dates(current_month,
                                                                                          path_to_preprocessed_files,
                                                                                          channel)
                missing_cropped_dates, existing_cropped_dates, target_data = find_missing_cropped_dates(current_month,
                                                                                                        path_to_cropped_files,
                                                                                                        channel)

                if np.all(missing_preprocessed_dates == missing_cropped_dates):
                    print('All possible images for {} successfully cropped.'.format(current_month.strftime('%Y/%m')))
                else:
                    print('Some existing preprocessed images were not successfully cropped:')
                    dates_to_check = missing_cropped_dates.difference(missing_preprocessed_dates)
                    collect_dates_to_check.extend(list(dates_to_check))

                current_month = current_month + relativedelta(months=1)
            print('All good?')
            print(len(collect_dates_to_check) == 0)
            print(collect_dates_to_check)

        return

    def _crop_images(self,channel, path_to_preprocessed_files,path_to_cropped_files,files_to_crop_names):
        n_cpus = cpu_count()
        print('Number of available CPUs:', n_cpus)
        print('Start cropping images.')
        Parallel(n_jobs=int(n_cpus / 2))(
            delayed(parallel_image_cropping)(file,path_to_preprocessed_files,path_to_cropped_files,self.downsample_resolution,
                                             self.crop_square_in_downsampled,self.resize_cropped) for file in files_to_crop_names)

        return


