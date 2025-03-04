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
from joblib import Parallel, delayed, cpu_count
from skimage.transform import rescale, resize
from skimage.measure import block_reduce
from matplotlib.patches import Circle
import cv2
from copy import copy
from utils import read_file_name
from astropy.time import Time


def register_AIA(smap, *, missing=None, order=3, method="scipy",arcsec_pix_target=None):
    """
    Processes a full-disk level 1 `~sunpy.map.sources.AIAMap` into a level
    1.5 `~sunpy.map.sources.AIAMap`.

    Rotates, scales and translates the image so that solar North is aligned
    with the y axis, each pixel is 0.6 arcsec across, and the center of the
    Sun is at the center of the image. The actual transformation is done by
    the `~sunpy.map.GenericMap.rotate` method.

    .. warning::

        This function might not return a 4096 by 4096 data array
        due to the nature of rotating and scaling the image.
        If you need a 4096 by 4096 image, you will need to pad the array manually,
        update header: crpix1 and crpix2 by the difference divided by 2 in size along that axis.
        Then create a new map.

        Please open an issue on the `aiapy GitHub page <https://github.com/LM-SAL/aiapy/issues>`__
        if you would like to see this changed.

    .. note::

        This routine modifies the header information to the standard
        ``PCi_j`` WCS formalism. The FITS header resulting in saving a file
        after this procedure will therefore differ from the original
        file.

    Parameters
    ----------
    smap : `~sunpy.map.sources.AIAMap` or `~sunpy.map.sources.sdo.HMIMap`
        A `~sunpy.map.Map` containing a full-disk AIA image or HMI magnetogram
    missing : `float`, optional
        If there are missing values after the interpolation, they will be
        filled in with ``missing``. If `None`, the default value will be the
        minimum value of ``smap``
    order : `int`, optional
        Order of the spline interpolation.
    method : {{{rotation_function_names}}}, optional
        Rotation function to use. Defaults to ``'scipy'``.

    Returns
    -------
    `~sunpy.map.sources.AIAMap` or `~sunpy.map.sources.sdo.HMIMap`:
        A level 1.5 copy of `~sunpy.map.sources.AIAMap` or
        `~sunpy.map.sources.sdo.HMIMap`.
    """
    # This implementation is taken directly from the `aiaprep` method in
    # sunpy.instr.aia.aiaprep under the terms of the BSD 2 Clause license.
    # See licenses/SUNPY.rst.

    orig_shape = smap.data.shape[0]

    # Target scale is 0.6 arcsec/pixel.
    if arcsec_pix_target is None:
        downsample_factor = 4096 / orig_shape
        arcsec_pix_target = 0.6 * downsample_factor

    if arcsec_pix_target is None and np.log2(smap.data.shape[0]) % 1 > 0:
        print('Warning: Map shape is not a power of 2. Please specify a target arcsec per pixel size.')

    scale = arcsec_pix_target * u.arcsec  # pragma: no cover # needs a full res image
    scale_factor = smap.scale[0] / scale
    missing = smap.min() if missing is None else missing
    tempmap = smap.rotate(
        recenter=True,
        scale=scale_factor.value,
        order=order,
        missing=missing,
        method=method,
    )
    # extract center from padded smap.rotate output
    # crpix1 and crpix2 will be equal (recenter=True), as prep does not work with submaps
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

def scale_AIA(smap, *, missing=None, order=3, method="scipy",rsun_target=976):


    orig_shape = smap.data.shape[0]

    scale_factor = rsun_target/smap.rsun_obs.value
    missing = smap.min() if missing is None else missing
    tempmap = smap.rotate(
        recenter=True,
        scale=scale_factor,
        order=order,
        missing=missing,
        method=method,
    )
    # extract center from padded smap.rotate output
    # crpix1 and crpix2 will be equal (recenter=True), as prep does not work with submaps
    center = np.floor(tempmap.meta["crpix1"])

    range_side = (center + np.array([-1, 1]) * smap.data.shape[0] / 2) * u.pix
    print(center)
    print(tempmap.data.shape[0])
    print(range_side)
    print(u.Quantity([range_side[0], range_side[0]]))
    print(u.Quantity([range_side[1], range_side[1]]) - 1 * u.pix)

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

def scale_image(img,rsun,rsun_target=976,missing=None, order=3):


    orig_shape = img.shape[0]

    scale_factor = rsun_target/rsun
    missing = img.min() if missing is None else missing

    tempimg = rescale(img, scale_factor, mode='constant',cval=missing,order=order)

    # extract center from padded rescale output
    # crpix1 and crpix2 will be equal (recenter=True), as prep does not work with submaps
    #crpix = tempimg.shape[0]/2 + 0.5
    #center = np.floor(crpix)
    #range_side = center + np.array([-1, 1]) * img.shape[0] / 2
    #print([range_side[0], range_side[0]])
    #print([range_side[1]-1, range_side[1]-1])

    print(tempimg.shape[0],orig_shape)
    if tempimg.shape[0] > orig_shape:
        center_pixel = tempimg.shape[0] / 2 - 0.5
        cutout_range = np.array([center_pixel - orig_shape / 2, center_pixel + orig_shape / 2])
        cutout_range[0] = cutout_range[0] + 0.5
        cutout_range[1] = cutout_range[1] + 0.5
        cutout_range = cutout_range.astype(int)
        newimg = tempimg[cutout_range[0]:cutout_range[1], cutout_range[0]:cutout_range[1]]
    elif tempimg.shape[0] < orig_shape:
        pad_width = int((orig_shape - tempimg.shape[0]) / 2)
        # Add padding
        newimg = np.pad(tempimg, pad_width, mode='constant', constant_values=0.)
    elif tempimg.shape[0] == orig_shape:
        newimg = tempimg

    return newimg


def register_HMI(smap, *, missing=None, order=3, method="scipy",arcsec_pix_target=None):
    # This implementation is taken directly from the `aiaprep` method in
    # sunpy.instr.aia.aiaprep under the terms of the BSD 2 Clause license.
    # See licenses/SUNPY.rst.

    orig_shape = smap.data.shape[0]
    # Target scale is 0.6 arcsec/pixel.
    if arcsec_pix_target is None:
        downsample_factor = 4096 / orig_shape
        arcsec_pix_target = 0.6 * downsample_factor

    if arcsec_pix_target is None and np.log2(smap.data.shape[0]) % 1 > 0:
        print('Warning: Map shape is not a power of 2. Please specify a target arcsec per pixel size.')

    scale = arcsec_pix_target * u.arcsec  # pragma: no cover # needs a full res image
    scale_factor = smap.scale[0] / scale

    tempmap = smap.rotate(
        recenter=True,
        scale=scale_factor.value,
        order=order,
        method=method,
        missing=0.0
    )

    # extract center from padded smap.rotate output
    # crpix1 and crpix2 will be equal (recenter=True), as prep does not work with submaps
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
        newmap_modified = big_map[cutout_range[0]:cutout_range[1], cutout_range[0]:cutout_range[1]]
        newmap = sunpy.map.Map(newmap_modified, newmap.meta)
        newmap.meta["crpix1"] = orig_shape / 2 + 0.5
        newmap.meta["crpix2"] = orig_shape / 2 + 0.5
    elif newmap.data.shape[0] < orig_shape:
        small_map = np.array(newmap.data)
        pad_width = int((orig_shape - small_map.shape[0]) / 2)
        # Add padding
        newmap_modified = np.pad(small_map, pad_width, mode='constant', constant_values=0.)
        newmap = sunpy.map.Map(newmap_modified, newmap.meta)
        newmap.meta["crpix1"] = orig_shape / 2 + 0.5
        newmap.meta["crpix2"] = orig_shape / 2 + 0.5


    return newmap


def preprocess_single_aia_image(aia_map,pointing_table,point_spread_function,correction_table,return_numpy=True,return_meta=False):
    # upsample to original dimension to perform pointing update and fix observer location
    new_dimensions = [4096, 4096] * u.pixel
    print('Updating pointing.')
    aia_resampled_map = aia_map.resample(new_dimensions)
    aia_map_updated_pointing = update_pointing(aia_resampled_map, pointing_table=pointing_table)
    #aia_map_observer_fixed = fix_observer_location(aia_map_updated_pointing)

    # then downsample map again to speed up deconvolution
    new_dimensions = [1024, 1024] * u.pixel
    #aia_map_observer_fixed_downsampled = aia_map_observer_fixed.resample(new_dimensions)
    aia_map_downsampled = aia_map_updated_pointing.resample(new_dimensions)

    print('Deconvolving.')
    # perform deconvolution and replace negative values in the corners of the image
    aia_map_deconvolved = deconvolve_bid(aia_map_downsampled.data, point_spread_function,use_gpu=False)
    aia_map_deconvolved[np.where(aia_map_deconvolved < 0)] = 0.0
    aia_map_deconvolved = sunpy.map.Map(aia_map_deconvolved, aia_map_downsampled.meta)

    print('Registering')
    # register map, i.e., rotate, scale and translate to align all images
    aia_map_registered = register_AIA(aia_map_deconvolved)
    rsun = aia_map_registered.rsun_obs.value
    wcs = aia_map_registered.wcs


    print('Correcting for degradation.')
    # correct for instrument degradation
    aia_map_degradation_corrected = correct_degradation(aia_map_registered, correction_table=correction_table)

    # normalize the image exposure time such that the units of the image are DN / pixel / s
    aia_map_exposure_time_corrected = aia_map_degradation_corrected.data / aia_map_degradation_corrected.exposure_time

    aia_map_numpy = np.array(aia_map_exposure_time_corrected.value)
    if return_numpy:
        if return_meta:
            return aia_map_numpy, wcs, rsun
        else:
            return aia_map_numpy


    aia_map_final = sunpy.map.Map(aia_map_numpy, aia_map_registered.meta)

    return aia_map_final

def preprocess_single_hmi_image(hmi_map, return_numpy=True, return_meta=False):
    '''
    import matplotlib.pyplot as plt
    # register map, i.e., rotate, scale and translate to align all images

    print(hmi_map.reference_pixel)
    print(hmi_map.scale)
    print(hmi_map.rotation_matrix)
    print(hmi_map.rsun_obs)
    print()

    fig = plt.figure()
    hmi_map.plot_settings['cmap'] = "hmimag"
    hmi_map.plot_settings['norm'] = plt.Normalize(-1500, 1500)
    ax = fig.add_subplot(projection=hmi_map)
    hmi_map.plot(axes=ax)
    hmi_map.draw_limb(axes=ax)
    '''

    #print(hmi_map.data.shape)

    hmi_map_registered = register_HMI(hmi_map)
    rsun = hmi_map_registered.rsun_obs.value
    wcs = hmi_map_registered.wcs

    map_aligned = np.array(hmi_map_registered.data)
    map_aligned_without_nan = np.nan_to_num(map_aligned)
    hmi_map_aligned_without_nan = sunpy.map.Map(map_aligned_without_nan, hmi_map_registered.meta)

    #print(hmi_map_registered.data.shape)

    '''
    print(hmi_map_aligned_without_nan.reference_pixel)
    print(hmi_map_aligned_without_nan.scale)
    print(hmi_map_aligned_without_nan.rotation_matrix)
    print(hmi_map_aligned_without_nan.rsun_obs)

    fig2 = plt.figure()
    hmi_map_aligned_without_nan.plot_settings['cmap'] = "hmimag"
    hmi_map_aligned_without_nan.plot_settings['norm'] = plt.Normalize(-1500, 1500)
    ax2 = fig2.add_subplot(projection=hmi_map_aligned_without_nan)
    hmi_map_aligned_without_nan.plot(axes=ax2)
    hmi_map_aligned_without_nan.draw_limb(axes=ax2)

    plt.show()

    exit()
    '''

    hmi_map_numpy = np.array(hmi_map_aligned_without_nan.data)
    if return_numpy:
        if return_meta:
            return hmi_map_numpy, wcs, rsun
        else:
            return hmi_map_numpy

    return hmi_map_registered


def parallel_aia_preprocessing(file,files_to_preprocess,path_to_input_files,path_to_output_month,pointing_table,psf_rebinned,correction_table):

    if file[-5:] == '.fits':
        fits_file = path_to_input_files + file
        # data has been downloaded from different sources and thus have different name patterns
        try:
            product = file[0:3]
            channel = file.split('.image')[0][-3:]
            date_str = file.split('T')[0][-10:]
            time_str = file.split('T')[1][:6]

            if int(time_str[-2:]) > 59:
                time_list = list(time_str)
                time_list[-2:] = '59'
                time_str = ''.join(time_list)
            date = datetime.strptime(date_str + '_' + time_str, '%Y-%m-%d_%H%M%S')
        except:
            product = file[0:3]
            channel = file.split('lev1_')[1][:3]
            date_str = file.split('t')[0][-10:]
            time_str = file.split('t')[1][:8]
            if int(time_str[-2:]) > 59:
                time_list = list(time_str)
                time_list[-2:] = '59'
                time_str = ''.join(time_list)
            date = datetime.strptime(date_str + '_' + time_str, '%Y_%m_%d_%H_%M_%S')
        # print(product, channel, date)
        try:
            aia_map = sunpy.map.Map(fits_file)
        except:
            print('Fits file could not be read.', product, channel, date)
            bad_quality = date
            meta = None
            return meta, bad_quality

        if not contains_full_disk(aia_map):
            print('Map does not contain full disk.', product, channel, date)
            bad_quality = date
            meta = None
            return meta, bad_quality
        #meta = None
        #bad_quality = None

        if aia_map.meta['QUALITY'] == 0:
            resolution = aia_map.data.shape[0]
            if resolution == 4096:
                aia_map = aia_map.resample([1024, 1024] * u.pixel)

            print('Processing', product, channel, date)

            preprocessed_aia_map, wcs, rsun = preprocess_single_aia_image(aia_map, pointing_table, psf_rebinned,
                                                                          correction_table, return_numpy=True,
                                                                          return_meta=True)

            #date_rounded = pd.Timestamp(date).round('60min').to_pydatetime()
            date_rounded = files_to_preprocess.loc[file]


            pickle.dump([preprocessed_aia_map, date, rsun, wcs],
                        open(path_to_output_month + channel + '_' + date_rounded.strftime('%Y-%m-%d_%H:%M') + '.pickle',
                             'wb'))
            #meta.append([date, rsun, wcs])
            meta = [date, rsun, wcs]
            bad_quality = None
        else:
            print('Bad quality', product, channel, date)
            #bad_quality.append(date)
            bad_quality = date
            meta = None

        return meta,bad_quality
    else:
        return None

def parallel_hmi_preprocessing(file,path_to_input_files,path_to_output_month,reproject=False,wcs_list_193=None,wcs_list_211=None):
    if file[-5:] == '.fits':
        fits_file = path_to_input_files + file

        product = file[0:3]
        date_str = file.split('720s.')[1][:8]
        time_str = file.split('_TAI')[0][-6:]
        if int(time_str[-2:]) > 59:
            time_list = list(time_str)
            time_list[-2:] = '59'
            time_str = ''.join(time_list)
        date = datetime.strptime(date_str + '_' + time_str, '%Y%m%d_%H%M%S')
        # print(product, channel, date)
        try:
            hmi_map = sunpy.map.Map(fits_file)
        except:
            print('Fits file could not be read.', product, 'magnetogram', date)
            bad_quality = date
            meta = None
            return meta, bad_quality

        if not contains_full_disk(hmi_map):
            print('Map does not contain full disk.', product, 'magnetogram', date)
            bad_quality = date
            meta = None
            return meta, bad_quality

        if hmi_map.meta['QUALITY'] == 0:
            resolution = hmi_map.data.shape[0]
            if resolution == 4096:
                hmi_map = hmi_map.resample([1024, 1024] * u.pixel)

            print('Processing', product, 'magnetogram', date)
            if reproject:


                within_1_hour_193 = (wcs_list_193.index + timedelta(hours=1) > date) & (
                            date > wcs_list_193.index - timedelta(hours=1))
                within_1_hour_211 = (wcs_list_211.index + timedelta(hours=1) > date) & (
                            date > wcs_list_211.index - timedelta(hours=1))
                wcs_193 = wcs_list_193[within_1_hour_193]
                wcs_211 = wcs_list_211[within_1_hour_211]

                wcs_found = True
                no_aia_wcs = None
                if len(wcs_193) == 1:
                    hmi_reprojected = hmi_map.reproject_to(wcs_193.iloc[0])
                elif len(wcs_211) == 1:
                    hmi_reprojected = hmi_map.reproject_to(wcs_211.iloc[0])
                elif len(wcs_193) > 1:
                    idx = np.argmin(np.abs(wcs_193.index - date))
                    hmi_reprojected = hmi_map.reproject_to(wcs_193.iloc[idx])
                elif len(wcs_211) > 1:
                    idx = np.argmin(np.abs(wcs_211.index - date))
                    hmi_reprojected = hmi_map.reproject_to(wcs_211.iloc[idx])
                else:
                    # no date found
                    wcs_found = False
                    #no_aia_wcs.append(date)
                    no_aia_wcs = date

                if wcs_found:
                    date_rounded = pd.Timestamp(date).round('60min').to_pydatetime()
                    hmi_reprojected = np.array(hmi_reprojected.data)
                    pickle.dump([hmi_reprojected, date],
                                open(path_to_output_month + 'hmi' + '_' + date_rounded.strftime(
                                    '%Y-%m-%d_%H:%M') + '.pickle',
                                     'wb'))
                    bad_quality = None

            else:
                hmi_preprocessed, wcs, rsun = preprocess_single_hmi_image(hmi_map,return_meta=True, return_numpy=True)

                date_rounded = pd.Timestamp(date).round('60min').to_pydatetime()

                pickle.dump([hmi_preprocessed, date, rsun, wcs],
                            open(path_to_output_month + 'hmi' + '_' + date_rounded.strftime('%Y-%m-%d_%H:%M') + '.pickle',
                                 'wb'))
                # meta.append([date, rsun, wcs])
                meta = [date, rsun, wcs]
                bad_quality = None
        else:
            print('Bad quality', product, 'magnetogram', date)
            # bad_quality.append(date)
            bad_quality = date
            meta = None

        return meta, bad_quality
    else:
        return None

def preprocess_sdo_data(start, end, path_to_raw, path_to_config, path_to_hmi, path_to_processed, wavelengths=None, preprocess_hmi=True):
    # first process
    if not os.path.isdir(path_to_processed):
        os.mkdir(path_to_processed)

    if wavelengths is None:
        wavelengths = [193, 211]


    ################################################################
    # AIA preprocessing
    ################################################################

    for wl in wavelengths:
        path_to_wl = path_to_raw.format(wl)

        print('Loading PSF.')
        try:
            print('Found saved PSF.')
            psf = pickle.load(open(path_to_config + 'psf_{}.pickle'.format(wl), 'rb'))
        except:
            print('Not found saved PSF. Compute it from scratch.')
            psf = aiapy.psf.psf(wl*u.angstrom)
            pickle.dump(psf, open(path_to_config + 'psf_{}.pickle'.format(wl), 'wb'))

        print('Rebinning PSF.')
        rebin_dimension = [1024, 1024]
        try:
            psf_rebinned = pickle.load(open(path_to_config + 'psf_{}_{}x{}.pickle'.format(wl, rebin_dimension[0], rebin_dimension[1]), 'rb'))
        except:
            psf_rebinned = rebin_psf(psf, rebin_dimension)
            pickle.dump(psf_rebinned, open(path_to_config + 'psf_{}_{}x{}.pickle'.format(wl, rebin_dimension[0], rebin_dimension[1]), 'wb'))

        print('Loading degradation correction table.')
        try:
            correction_table = pickle.load(open(path_to_config + 'degradation_correction_table.pickle', 'rb'))
        except:
            correction_table = get_correction_table("JSOC")
            pickle.dump(correction_table, open(path_to_config + 'degradation_correction_table.pickle', 'wb'))


        path_to_output = path_to_processed + '/aia_{}'.format(str(wl))
        if not os.path.isdir(path_to_output):
            os.mkdir(path_to_output)

        current_month = datetime(start.year,start.month,1)

        while current_month <= end:
            
            month_str = current_month.strftime('%Y/%m')
            path_to_input_files = path_to_wl + month_str + '/'

            path_to_output_year = path_to_output + '/' + current_month.strftime('%Y')
            if not os.path.isdir(path_to_output_year):
                os.mkdir(path_to_output_year)
            path_to_output_month = path_to_output_year + '/' + current_month.strftime('%m')
            if not os.path.isdir(path_to_output_month):
                os.mkdir(path_to_output_month)
            path_to_output_month = path_to_output_month + '/'

            print('Loading pointing table for {}.'.format(month_str))
            try:
                pointing_table = pickle.load(open(path_to_config + 'pointing_table_{}.pickle'.format(current_month.strftime('%Y%m')), 'rb'))
            except:
                #print(Time(current_month-timedelta(days=1)),Time(current_month+timedelta(days=32)))
                time_range = (Time(current_month-timedelta(days=1)),Time(current_month+timedelta(days=32)))
                pointing_table = aiapy.calibrate.util.get_pointing_table("JSOC", time_range=time_range)
                pickle.dump(pointing_table, open(path_to_config + 'pointing_table_{}.pickle'.format(current_month.strftime('%Y%m')), 'wb'))


            #########################################
            month_start = datetime(current_month.year,current_month.month,1,0)
            month_end = month_start + relativedelta(months=1)-timedelta(hours=1)
            target_data = pd.date_range(month_start, month_end, freq='1h')

            output_files = os.listdir(path_to_output_month)
            output_files.sort()
            existing_dates = []
            for file in output_files:
                if file[-7:] == '.pickle' and file[:3] == str(wl):
                    file_date, _, _ = read_file_name(file,preprocessed=True)
                    existing_dates.append(file_date)
            existing_dates = pd.Index(existing_dates)

            missing_dates = target_data.difference(existing_dates)
            print('Missing dates that still need to be preprocessed:')
            print(missing_dates)

            try:
                substitute_dates = pd.read_csv(path_to_input_files + 'substitute_dates.csv',index_col=0,parse_dates=[0,1]).squeeze()
                print('Substitute dates found.')
                # check if there are duplicates
                duplicate_mask = substitute_dates.index.duplicated(keep='first')
                FLAG = False
                if np.sum(duplicate_mask) > 0:
                    FLAG=True
                duplicate_indices = substitute_dates.index[duplicate_mask]
                print('Substitute dates before replacement:')
                print(len(substitute_dates))
                print('Duplicate indices:',duplicate_indices)
                for entry in duplicate_indices:
                    print('Looking for replacement:', entry)
                    duplicate_entries = substitute_dates.loc[entry].values
                    print('Available values:', duplicate_entries)
                    found_valid = False
                    for val in duplicate_entries:
                        if not pd.isnull(val):
                            valid_date = copy(val)
                            substitute_dates.loc[entry] = valid_date
                            found_valid = True
                            print('Found replacement:',entry,valid_date)
                            break
                    if not found_valid:
                        print('No replacement found:',entry)

                substitute_dates = substitute_dates.loc[np.invert(duplicate_mask)]
                print('Substitute dates after replacement:')
                print(len(substitute_dates))

            except:
                substitute_dates = None
                print('Substitute dates were not found.')

            #substitute_dates = None
            print(substitute_dates)

            input_files = os.listdir(path_to_input_files)
            input_files.sort()
            input_file_dates_exact = []
            input_file_dates_rounded = []
            input_file_names = []
            for file in input_files:
                if file[-5:] == '.fits':
                    file_date, _, _ = read_file_name(file)
                    input_file_names.append(file)
                    input_file_dates_exact.append(file_date)
                    input_file_dates_rounded.append(pd.Timestamp(file_date).round('5min').to_pydatetime())
            available_input_files_exact = pd.Series(input_file_names,index=input_file_dates_exact)
            available_input_files_rounded = pd.Series(input_file_names,index=input_file_dates_rounded) # could have index duplicates


            # find for each missing date an input file to fill that gap
            files_to_preprocess = []
            files_to_preprocess_date = []

            new_missing_dates = []
            double_dates = []


            for missing_date in missing_dates:
                if substitute_dates is not None and missing_date in substitute_dates.index:
                    file_date = substitute_dates.loc[missing_date]

                    if pd.isnull(file_date):
                        new_missing_dates.append(missing_date)
                    else:
                        # find filename
                        file_name = available_input_files_exact.loc[file_date]
                        files_to_preprocess.append(file_name)
                        files_to_preprocess_date.append(missing_date)
                else:
                    mask = available_input_files_rounded.index == missing_date
                    if np.sum(mask) == 1:
                        file_name = available_input_files_rounded.loc[mask].squeeze()
                        files_to_preprocess.append(file_name)
                        files_to_preprocess_date.append(missing_date)
                    elif np.sum(mask) > 1:
                        print('That should not happen!')
                        file_names = list(available_input_files_rounded.loc[mask])
                        print(file_names)
                        files_to_preprocess.extend(file_names)
                        files_to_preprocess_date.extend([missing_date] * np.sum(mask))
                        double_dates.append(missing_date)
                    elif np.sum(mask) == 0:
                        new_missing_dates.append(missing_date)

            files_to_preprocess = pd.Series(files_to_preprocess_date,index=files_to_preprocess)


            meta = []
            bad_quality = []

            #res = Parallel(n_jobs=-1)(delayed(parallel_aia_preprocessing)(file,files_to_preprocess,path_to_input_files,path_to_output_month,pointing_table,psf_rebinned,correction_table) for file in list(files_to_preprocess))
            n_cpus = cpu_count()
            print('Number of available CPUs:', n_cpus)

            res = Parallel(n_jobs=int(n_cpus/2))(delayed(parallel_aia_preprocessing)(file,files_to_preprocess,path_to_input_files,path_to_output_month,pointing_table,psf_rebinned,correction_table) for file in list(files_to_preprocess.index))
            for item in res:
                if item is not None:
                    meta_item = item[0]
                    if meta_item is not None:
                        meta.append(meta_item)
                    bad_quality_item = item[1]
                    if bad_quality_item is not None:
                        bad_quality.append(bad_quality_item)
            print(meta,bad_quality)

            meta = np.array(meta)
            bad_quality = np.array(bad_quality)

            try:
                meta_old = pickle.load(meta, open(path_to_output_month + 'meta.pickle', 'rb'))
                meta = np.concatenate([meta_old,meta],axis=0)
            except:
                pass
            pickle.dump(meta,open(path_to_output_month + 'meta.pickle', 'wb'))

            try:
                bad_quality_old = pickle.load(meta, open(path_to_output_month + 'bad_quality_dates.pickle', 'rb'))
                bad_quality = np.concatenate([bad_quality_old,bad_quality],axis=0)
            except:
                pass
            pickle.dump(bad_quality,open(path_to_output_month + 'bad_quality_dates.pickle', 'wb'))


            current_month = current_month + relativedelta(months=1)


    ################################################################
    # HMI magnetogram preprocessing
    ################################################################
    if preprocess_hmi:
        path_to_wl = path_to_hmi + '/'

        path_to_output = path_to_processed + '/hmi'
        if not os.path.isdir(path_to_output):
            os.mkdir(path_to_output)

        current_month = datetime(start.year, start.month, 1)

        while current_month <= end:
            month_str = current_month.strftime('%Y/%m')
            path_to_input_files = path_to_wl + month_str + '/'

            path_to_output_year = path_to_output + '/' + current_month.strftime('%Y')
            if not os.path.isdir(path_to_output_year):
                os.mkdir(path_to_output_year)
            path_to_output_month = path_to_output_year + '/' + current_month.strftime('%m')
            if not os.path.isdir(path_to_output_month):
                os.mkdir(path_to_output_month)
            path_to_output_month = path_to_output_month + '/'

            files = os.listdir(path_to_input_files)
            files.sort()

            # load WCS metadata from AIA map
            #aia_193_meta = pickle.load(open(path_to_processed + '/aia_193/' + month_str + '/meta.pickle', 'rb'))
            #wcs_list_193 = None # pd.Series(aia_193_meta[:, 2], index=aia_193_meta[:, 0])

            #aia_211_meta = pickle.load(open(path_to_processed + '/aia_211/' + month_str + '/meta.pickle', 'rb'))
            #wcs_list_211 = None # pd.Series(aia_211_meta[:, 2], index=aia_211_meta[:, 0])

            #raise ValueError('AIA meta data not found. It is needed to reproject HMI map onto AIA map. Please preprocess the AIA data first and then specify the path to the meta data.')

            bad_quality = []
            meta = []
            print('Number of available CPU cores:',cpu_count())
            res = Parallel(n_jobs=-1)(
                delayed(parallel_hmi_preprocessing)(file,path_to_input_files,path_to_output_month) for file in files)
            for item in res:
                if item is not None:
                    meta_item = item[0]
                    if meta_item is not None:
                        meta.append(meta_item)
                    bad_quality_item = item[1]
                    if bad_quality_item is not None:
                        bad_quality.append(bad_quality_item)
            print(meta, bad_quality)

            '''
            for file in files:
                if file[-5:] == '.fits':
                    fits_file = path_to_input_files + file
    
                    product = file[0:3]
                    date_str = file.split('720s.')[1][:8]
                    time_str = file.split('_TAI')[0][-6:]
    
                    date = datetime.strptime(date_str + '_' + time_str, '%Y%m%d_%H%M%S')
                    # print(product, channel, date)
    
                    hmi_map = sunpy.map.Map(fits_file)
    
    
                    if hmi_map.meta['QUALITY'] == 0:
                        resolution = hmi_map.data.shape[0]
                        if resolution == 4096:
                            hmi_map = hmi_map.resample([1024, 1024] * u.pixel)
    
                        print('Processing', product, 'magnetogram', date)
                        #date_rounded = pd.Timestamp(date).round('60min').to_pydatetime()
    
                        within_1_hour_193 = (wcs_list_193.index + timedelta(hours=1) > date) & (date > wcs_list_193.index - timedelta(hours=1))
                        #within_1_hour_211 = (wcs_list_211.index + timedelta(hours=1) > date) & (
                        #            date > wcs_list_211.index - timedelta(hours=1))
                        wcs_193 = wcs_list_193[within_1_hour_193]
                        #wcs_211 = wcs_list_211[within_1_hour_211]
    
                        wcs_found = True
                        if len(wcs_193) == 1:
                            hmi_reprojected = hmi_map.reproject_to(wcs_193.iloc[0])
                        #elif len(wcs_211) == 1:
                        #    hmi_reprojected = hmi_map.reproject_to(wcs_211.iloc[0])
                        elif len(wcs_193) > 1:
                            idx = np.argmin(np.abs(wcs_193.index - date))
                            hmi_reprojected = hmi_map.reproject_to(wcs_193.iloc[idx])
                        #elif len(wcs_211) > 1:
                        #    idx = np.argmin(np.abs(wcs_211.index - date))
                        #    hmi_reprojected = hmi_map.reproject_to(wcs_211.iloc[idx])
                        else:
                            # no date found
                            wcs_found = False
                            no_aia_wcs.append(date)
    
                        if wcs_found:
                            date_rounded = pd.Timestamp(date).round('60min').to_pydatetime()
                            hmi_reprojected = np.array(hmi_reprojected.data)
                            pickle.dump([hmi_reprojected, date],
                                        open(path_to_output_month + 'hmi' + '_' + date_rounded.strftime('%Y-%m-%d_%H:%M') + '.pickle',
                                             'wb'))
                    else:
                        print('Bad quality', product, 'magnetogram', date)
                        bad_quality.append(date)
            '''

            pickle.dump(np.array(bad_quality),
                        open(path_to_output_month + 'bad_quality_dates.pickle', 'wb'))
            pickle.dump(np.array(meta),
                        open(path_to_output_month + 'meta.pickle', 'wb'))

            current_month = current_month + relativedelta(months=1)
    return


def parallel_aia_cropping(file,path_to_input_files,path_to_output_month):
    if file[-7:] == '.pickle':
        #try:
        img,date,rsun,wcs = pickle.load(open(path_to_input_files + file,'rb'))
        print(date)

        # data is provided upside-down, so it needs to be flipped such that the y-axis is aligned to solar north
        img = np.flipud(img)
        img = scale_image(img, rsun)
        # downsample by summing in blocks
        img = block_reduce(img, (2, 2), np.sum)
        # crop image to 300x300 pixels
        cut_pixels = int((img.shape[0]-300)/2)
        img = img[cut_pixels:-cut_pixels,cut_pixels:-cut_pixels]
        # downsample image to 224x224 pixels
        #fig3,ax3=plt.subplots()
        #ax3.imshow(img,cmap='jet')
        #circle1 = Circle((149.5, 149.5), 976 / 4.8, color='white', fill=False, linewidth=1)
        #ax3.add_patch(circle1)
        img = resize(img,(224,224),order=3,mode='constant',cval=0)

        img = img.astype('float32')
        np.save(path_to_output_month.format('numpy') + file[:-7] + '.npy', img)

        img_up = normalize_image_upendran(copy(img),193)
        cv2.imwrite(path_to_output_month.format('norm_upendran') + file[:-7] + '.jpg', img_up)

        img_ja = normalize_image_jarolim(copy(img),193)
        cv2.imwrite(path_to_output_month.format('norm_jarolim') + file[:-7] + '.jpg', img_ja)
        #except:
        #    print('Not able to load file!')
    return

def normalize_image_upendran(img,wavelength=193):
    if wavelength == 193:
        mask = img < 125
        img[mask] = 125
        mask = img > 5000
        img[mask] = 5000
        img = np.log(img)
        # Normalize to [0, 255] (Min-Max Scaling)
        min_val, max_val = img.min(), img.max()
        img_normalized = 255 * (img - min_val) / (max_val - min_val)
    elif wavelength == 211:
        mask = img < 25
        img[mask] = 25
        mask = img > 2500
        img[mask] = 2500
        img = np.log(img)
        # Normalize to [0, 255] (Min-Max Scaling)
        min_val, max_val = img.min(), img.max()
        img_normalized = 255 * (img - min_val) / (max_val - min_val)
    img = img_normalized.astype(np.uint8)  # Convert to uint8


    return img

def normalize_image_jarolim(img,wavelength=193):
    a = 0.005
    lower_limit = 0
    if wavelength == 193:
        upper_limit = 7757.31
    elif wavelength == 211:
        upper_limit = 6539
    # add magnetogram

    mask = img < lower_limit
    img[mask] = lower_limit
    mask = img > upper_limit
    img[mask] = upper_limit

    # Scale to [0, 1] (Min-Max Scaling)
    min_val, max_val = img.min(), img.max()
    img = 1 * (img - min_val) / (max_val - min_val)

    img = np.asinh(img/a)/np.asinh(1/a)

    # Normalize to [0, 255] (Min-Max Scaling)
    min_val, max_val = img.min(), img.max()
    img_normalized = 255 * (img - min_val) / (max_val - min_val)
    img = img_normalized.astype(np.uint8)  # Convert to uint8

    return img


dir_path = os.path.dirname(os.getcwd())

start = datetime(2010,5,1)
end = datetime(2024,12,31,23)
run_on_server = True
wavelengths = [193]
preprocess_hmi = False

if run_on_server:
    path_to_hmi = dir_path + '/data/SDO/HMI/magnetogram'
    path_to_raw = dir_path + '/data/SDO/AIA/{}/'
    path_to_preprocessed = dir_path + '/processed_data/deep_learning'
    path_to_config = dir_path + '/data/configuration_data/'
else:
    path_to_hmi = dir_path + '/data/raw_data/'
    path_to_raw = dir_path + '/data/raw_data/aia_{}/'
    path_to_preprocessed = dir_path + '/data/processed_data/deep_learning'
    path_to_config = dir_path + '/data/configuration_data/'


preprocess_sdo_data(start,end,path_to_raw,path_to_config,path_to_hmi,path_to_preprocessed,wavelengths,preprocess_hmi)
exit()
path_to_aia = dir_path + '/SDO/AIA'
path_to_hmi = dir_path + '/SDO/HMI/magnetogram'
#path_to_preprocessed = dir_path + '/images_preprocessed'

'''
start = datetime(2010,6,1)
end = datetime(2010,6,30,23)
path_to_aia = dir_path + '/data/AIA'
path_to_hmi = dir_path + '/data/HMI'
path_to_preprocessed = dir_path + '/data/images_preprocessed'
'''

wavelengths = [193]
preprocess_hmi = False



path_to_preprocessed = dir_path + '/processed_data/deep_learning/aia_{}/{}/{}/'
path_to_final = dir_path + '/processed_data/dl_cropped_brown/'

# create folders
for name in ['numpy','norm_upendran','norm_jarolim']:
    if not os.path.isdir(path_to_final + name):
        os.mkdir(path_to_final + name)
    for wl in wavelengths:
        if not os.path.isdir(path_to_final + name + '/aia_' + str(wl)):
            os.mkdir(path_to_final + name + '/aia_' + str(wl))

        current_month = start

        while current_month < end:
            if not os.path.isdir(path_to_final + name + '/aia_' + str(wl) + '/' + current_month.strftime('%Y')):
                os.mkdir(path_to_final + name + '/aia_' + str(wl) + '/' + current_month.strftime('%Y'))

            if not os.path.isdir(path_to_final + name + '/aia_' + str(wl) + '/' + current_month.strftime('%Y') + '/' + current_month.strftime('%m')):
                os.mkdir(path_to_final + name + '/aia_' + str(wl) + '/' + current_month.strftime('%Y') + '/' + current_month.strftime('%m'))

            current_month = current_month + relativedelta(months=1)


for wl in wavelengths:

    current_month = start

    while current_month < end:

        path_input = path_to_preprocessed.format(wl, current_month.strftime('%Y'), current_month.strftime('%m'))

        path_output = path_to_final + '{}/aia_' + str(wl) + '/' + current_month.strftime('%Y') + '/' + current_month.strftime('%m') + '/'

        files = os.listdir(path_input)
        files.sort()
        files.remove('meta.pickle')
        files.remove('bad_quality_dates.pickle')
        print('Number of available CPUs:', cpu_count())
        res = Parallel(n_jobs=-1)(
            delayed(parallel_aia_cropping)(file, path_input, path_output) for file in files)

        current_month = current_month + relativedelta(months=1)


print('SUCCESS!!!')
exit()
#print(files)
'''
from astropy.time import Time
img_ref = sunpy.map.Map('/home/collin/projects/distributional_regression/solar_images/data/raw_data/aia_193/2015/06/aia_lev1_193a_2015_06_01t00_00_06_84z_image_lev1.fits')

resolution = img_ref.data.shape[0]
print(resolution)
if resolution == 4096:
    img_ref = img_ref.resample([1024, 1024] * u.pixel)
resolution = img_ref.data.shape[0]
print(resolution)
st = Time(datetime(2015,6,1) - timedelta(days=3),format='datetime')
en = Time(datetime(2015,6,1) + timedelta(days=3),format='datetime')

pointing_table = aiapy.calibrate.util.get_pointing_table('jsoc',time_range=(st,en))
#psf = pickle.load(open(dir_path + '/data/point_spread_functions/psf_{}.pickle'.format(193), 'rb'))
print('Rebinning PSF.')
#rebin_dimension = [1024, 1024]
#psf_rebinned = rebin_psf(psf, rebin_dimension)
#pickle.dump(psf_rebinned,open(dir_path + '/data/point_spread_functions/psf_{}_1024.pickle'.format(193), 'wb'))
psf_rebinned = pickle.load(open(dir_path + '/data/point_spread_functions/psf_{}_1024.pickle'.format(193), 'rb'))

print('Loading degradation correction table.')
correction_table = get_correction_table('jsoc')

preprocessed_img_ref = preprocess_single_aia_image(img_ref, pointing_table, psf_rebinned,
                                                              correction_table, return_numpy=False,
                                                              return_meta=False)
pickle.dump(preprocessed_img_ref,open('img_ref_pre.pickle','wb'))
exit()
'''
'''
img_ref = pickle.load(open('img_ref_pre.pickle','rb'))
img_ref = sunpy.map.Map(img_ref)
print(img_ref.rsun_obs)

print(img_ref.reference_pixel)
print(img_ref.scale)
print(img_ref.data.shape)
print(img_ref.rsun_obs.value/img_ref.scale[0].value)
#img_ref.peek(clip_interval=(1, 99.99) * u.percent, draw_limb=True)

rsun_target = 976
#img_ref.peek(draw_limb=True)
img_ref_scaled = scale_AIA(img_ref)
print(img_ref_scaled.rsun_obs)
print(img_ref_scaled.reference_pixel)
print(img_ref_scaled.scale)
print(img_ref_scaled.data.shape)
print(img_ref_scaled.rsun_obs.value/img_ref_scaled.scale[0].value)
fig,ax = plt.subplots()
ax.imshow(img_ref.data)
c1 = Circle((511.5,511.5),394.3886204166667,fill=False)
ax.add_patch(c1)
#new_img = rescale(img_ref.data, rsun_target/img_ref_scaled.rsun_obs.value)
new_img = scale_image(img_ref.data, img_ref.rsun_obs.value)

fig1,ax1 = plt.subplots()
ax1.imshow(new_img)
c2 = Circle((511.5,511.5),406.6666666666667,fill=False)
ax1.add_patch(c2)

fig2,ax2 = plt.subplots()
dif = img_ref_scaled.data - new_img
print(dif.min(),dif.max(),dif.mean(),np.abs(dif).mean())
ax2.imshow(dif)
#plt.show()
#img_ref_scaled.peek(draw_limb=True)
'''

# muss geflippt werden?
# radius anpassen
# zu jpg
# downsampling
# cutout
# magnetograms auch auf gleichen radius


data = pickle.load(open(path_to_preprocessed + 'meta.pickle','rb'))
dates = data[:,0]
rsun = data[:,1]

#plt.plot(dates,rsun)
#plt.show()

# after preprocessing: reference_pixel = (511.5,511.5) (center of solar disk in imshow)

files.remove('meta.pickle')



data = pickle.load(open(path_to_preprocessed + 'bad_quality_dates.pickle','rb'))
print(data)
files.remove('bad_quality_dates.pickle')

dates = []
rsuns = []
import cv2
from copy import copy
path_to_normalized = dir_path + '/data/processed_data/normalized/'

for file in files:
    #print(file)
    if file[-7:] == '.pickle':
        img,date,rsun,wcs = pickle.load(open(path_to_preprocessed + file,'rb'))
        print(date)
        dates.append(date)
        rsuns.append(rsun)
        # data is provided upside-down, so it needs to be flipped such that the y-axis is aligned to solar north
        img = np.flipud(img)
        img = scale_image(img, rsun)
        # downsample by summing in blocks
        img = block_reduce(img, (2, 2), np.sum)
        # crop image to 300x300 pixels
        cut_pixels = int((img.shape[0]-300)/2)
        img = img[cut_pixels:-cut_pixels,cut_pixels:-cut_pixels]
        # downsample image to 224x224 pixels
        #fig3,ax3=plt.subplots()
        #ax3.imshow(img,cmap='jet')
        #circle1 = Circle((149.5, 149.5), 976 / 4.8, color='white', fill=False, linewidth=1)
        #ax3.add_patch(circle1)
        img = resize(img,(224,224),order=3,mode='constant',cval=0)

        np.save(path_to_normalized + 'numpy/' + file[:-7] + '.npy',img)

        img_up = normalize_image_upendran(copy(img),193)
        cv2.imwrite(path_to_normalized + 'upendran/' + file[:-7] + '.jpg', img_up)

        img_ja = normalize_image_jarolim(copy(img),193)
        cv2.imwrite(path_to_normalized + 'jarolim/' + file[:-7] + '.jpg', img_ja)

        img = img.astype('float32')
        np.save(path_to_normalized + 'numpy32/' + file[:-7] + '.npy', img)


        #img = np.flipud
        #fig,ax = plt.subplots()
        #ax.imshow(img,cmap='jet')
        #plt.show()
        '''
        circle2 = Circle((img_ref.reference_pixel[0].value,img_ref.reference_pixel[1].value), img_ref.rsun_obs.value/img_ref.scale[0].value, color='red', fill=False, linewidth=1)
        ax.add_patch(circle2)

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(projection=img_ref)
        img_ref.plot(axes=ax1)
        img_ref.draw_limb(axes=ax1)
        circle1 = Circle((511.5, 511.5), rsun / 2.4, color='blue', fill=False, linewidth=1)
        ax1.add_patch(circle1)
        circle2 = Circle((img_ref.reference_pixel[0].value, img_ref.reference_pixel[1].value),
                         img_ref.rsun_obs.value / img_ref.scale[0].value, color='red', fill=False, linewidth=1)
        ax1.add_patch(circle2)

        plt.show()

        # ax1 = fig.add_subplot(121, projection=img_ref)
        # img_ref.plot(axes=ax1)
        # img_ref.draw_limb(axes=ax1)

        img_ref.peek(clip_interval=(1, 99.99) * u.percent,draw_limb=True)


        plt.show()
        '''

plt.plot(dates,rsuns)
plt.show()
#preprocess_sdo_data(start,end,path_to_aia,path_to_hmi,path_to_preprocessed,wavelengths,preprocess_hmi)