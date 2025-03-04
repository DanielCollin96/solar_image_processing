import jsoc_download as jd
from datetime import datetime,timedelta
from dateutil.relativedelta import relativedelta
import os
import numpy as np
import pandas as pd
import pickle
from copy import copy
import time
#import skimage
import astropy
from rebin_psf import rebin_psf
from deconvolve_image import deconvolve_bid

import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
from astropy.io import fits
import aiapy.psf
from aiapy.calibrate import register, update_pointing, fix_observer_location, correct_degradation
from aiapy.calibrate.util import get_correction_table
import matplotlib.pyplot as plt
import sunpy.map
import astropy.units as u
import aiapy.data.sample as sample_data
from aiapy.calibrate import fix_observer_location, update_pointing
from aiapy.calibrate import register, update_pointing
from astropy.coordinates import SkyCoord
from astropy.visualization import AsinhStretch, ImageNormalize, LogStretch
from copy import copy,deepcopy
#from skimage import transform
from sunpy.coordinates import Helioprojective, propagate_with_solar_surface
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS







def preprocess_single_aia_image(aia_map,pointing_table,point_spread_function,correction_table,return_numpy=True,return_rsun=False):

    # upsample to original dimension to perform pointing update and fix observer location
    new_dimensions = [4096, 4096] * u.pixel
    aia_resampled_map = aia_map.resample(new_dimensions)

    aia_map_updated_pointing = update_pointing(aia_resampled_map, pointing_table=pointing_table)
    aia_map_observer_fixed = fix_observer_location(aia_map_updated_pointing)

    # then downsample map again to speed up deconvolution
    new_dimensions = [1024, 1024] * u.pixel
    aia_map_observer_fixed_downsampled = aia_map_observer_fixed.resample(new_dimensions)

    # perform deconvolution and replace negative values in the corners of the image
    aia_map_deconvolved = deconvolve_bid(aia_map_observer_fixed_downsampled.data, point_spread_function, use_gpu=False)

    aia_map_deconvolved[np.where(aia_map_deconvolved < 0)] = 0.0
    aia_map_deconvolved = sunpy.map.Map(aia_map_deconvolved, aia_map_observer_fixed_downsampled.meta)

    # register map, i.e., rotate, scale and translate to align all images
    aia_map_registered = register_AIA(aia_map_deconvolved)
    rsun = aia_map_registered.rsun_obs.value

    # correct for instrument degradation
    aia_map_degradation_corrected = correct_degradation(aia_map_registered, correction_table=correction_table)

    # normalize the image exposure time such that the units of the image are DN / pixel / s
    aia_map_exposure_time_corrected = aia_map_degradation_corrected.data / aia_map_degradation_corrected.exposure_time

    aia_map_numpy = np.array(aia_map_exposure_time_corrected.value)
    if return_numpy:
        if return_rsun:
            return aia_map_numpy, rsun
        else:
            return aia_map_numpy


    aia_map_final = sunpy.map.Map(aia_map_numpy, aia_map_registered.meta)

    return aia_map_final


# look for projection_wcs in
def preprocess_single_hmi_image(projection_wcs):
    # check if corresponding AIA image does already exist
    # if not, raise error

    # check for dimension. Downsample to 1024x1024 if bigger

    # check quality key
    # if yes, load it
    return

def preprocess_sdo_data(start, end, path_to_aia, path_to_hmi, path_to_processed, wavelengths=None):
    # first process

    if wavelengths is None:
        wavelengths = [193, 211]
    path_to_aia = '/home/collin/projects/distributional_regression/data/AIA/193/2010/06/'
    path_to_processed = '/home/collin/projects/distributional_regression/data/preprocessed/aia_193/2010/06/'

    for wl in wavelengths:

        path_to_raw = path_to_aia + '/' + str(wl) + '/'
        psf = aiapy.psf.psf(wl)
        #psf = pickle.load(open('psf_193.pickle', 'rb'))
        rebin_dimension = [1024, 1024]
        psf_rebinned = rebin_psf(psf, rebin_dimension)
        correction_table = get_correction_table()

        path_to_output = path_to_processed + '/aia_{}' + format(str(wl)) + '/'
        current_month = datetime(start.year,start.month,1)

        while current_month <= end:
            month_str = current_month.strftime('%Y/%m')
            path_to_input_files = path_to_raw + month_str + '/'
            path_to_output_files = path_to_output + month_str + '/'

            pointing_table = aiapy.calibrate.util.get_pointing_table(current_month-timedelta(days=1), current_month+timedelta(days=32))


            files = os.listdir(path_to_input_files)
            files.sort()
            for file in files:
                if file[-5:] == '.fits':
                    fits_file = path_to_input_files + file
                    product = file[0:3]
                    channel = file.split('.image')[0][-3:]
                    date_str = file.split('T')[0][-10:]
                    time_str = file.split('T')[1][:6]

                    date = datetime.strptime(date_str + '_' + time_str, '%Y-%m-%d_%H%M%S')
                    #print(product, channel, date)

                    aia_map = sunpy.map.Map(fits_file)

                    bad_quality = []
                    if aia_map.meta['QUALITY'] == 0:
                        resolution = aia_map.data.shape[0]
                        if resolution == 4096:
                            aia_map = aia_map.resample([1024,1024]*u.pixel)

                        print('Processing', product, channel, date)
                        preprocessed_aia_map, rsun = preprocess_single_aia_image(aia_map, pointing_table, psf_rebinned, correction_table,return_numpy=True,return_rsun=True)

                        pickle.dump([preprocessed_aia_map, date, rsun],
                                    open(path_to_output_files + channel + '_' + date.strftime('%Y-%m-%d_%H:%M') + '.pickle', 'wb'))
                    else:
                        print('Bad quality', product, channel, date)
                        bad_quality.append(date)
                    pickle.dump(bad_quality,
                                open(path_to_processed + 'bad_quality_dates.pickle', 'wb'))
    return



def manual_registering():
    arcsec_pix_actual = (aia_map_observer_fixed.scale[0].value + aia_map_observer_fixed.scale[1].value) / 2
    arcsec_pix_target = 0.6
    scaling_factor = arcsec_pix_actual / arcsec_pix_target
    print(scaling_factor)
    anti_aliasing = False
    if scaling_factor < 1:
        anti_aliasing = True
    trans = transform.rescale(trans, preserve_range=True, scale=scaling_factor, anti_aliasing=anti_aliasing)
    print(trans.shape)

    ###########################################
    # Given rotation matrix
    rot_matrix = aia_map_observer_fixed.rotation_matrix
    # Calculate the rotation angle in radians
    rot_angle = np.degrees(np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0]))
    trans = transform.rotate(trans, rot_angle, preserve_range=True)
    print(trans.shape)

    missing = aia_map_observer_fixed.min()
    tempmap = aia_map_observer_fixed.rotate(
        recenter=True,
        scale=scaling_factor,
        order=3,
        missing=missing,
        method='scikit-image',
    )
    print(tempmap.data.shape)
    dif = np.abs(tempmap.data - trans)
    print(np.mean(dif), np.max(dif))

    ###########################################
    if trans.shape[0] > orig_shape[0]:
        new_center_pixel = trans.shape[0] / 2 - 0.5
        cutout_range = np.array([new_center_pixel - orig_shape[0] / 2, new_center_pixel + orig_shape[0] / 2])
        cutout_range[0] = cutout_range[0] + 0.5
        cutout_range[1] = cutout_range[1] + 0.5
        cutout_range = cutout_range.astype(int)
        trans = trans[cutout_range[0]:cutout_range[1], cutout_range[0]:cutout_range[1]]
    elif trans.shape[0] < orig_shape[0]:
        pad_width = int((orig_shape[0] - trans.shape[0]) / 2)
        # Add padding
        trans = np.pad(trans, pad_width, mode='constant', constant_values=0.)
    # Given rotation matrix
    rot_matrix = aia_map_observer_fixed.rotation_matrix
    # Calculate the rotation angle in radians
    rot_angle = np.degrees(np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0]))
    trans = transform.rotate(trans, rot_angle, preserve_range=True)
    print(trans.shape)

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

    # Target scale is 0.6 arcsec/pixel.
    if arcsec_pix_target is None:
        downsample_factor = 4096 / smap.data.shape[0]
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
    return newmap

def test_registration_functions():

    ###########
    sample_img = np.arange(100*100).reshape((100,100))
    orig_shape = np.array(sample_img.shape)
    sample_img[22,33] = 20000
    center_pixel_actual = np.array([22,33])
    center_pixel_desired = sample_img.shape[0]/2 - 0.5
    center_pixel_desired = [center_pixel_desired,center_pixel_desired]
    print(center_pixel_actual,center_pixel_desired)
    diff = center_pixel_actual - center_pixel_desired
    print(diff)
    plt.imshow(sample_img)
    plt.show()
    tform = transform.SimilarityTransform(translation=(diff[1],diff[0]))
    trans = transform.warp(sample_img,tform,preserve_range=True)
    t = plt.imshow(trans)
    plt.colorbar(t)
    plt.show()
    scale=0.9
    anti_aliasing = False
    if scale < 1:
        anti_aliasing = True

    trans = transform.rescale(trans,preserve_range=True,scale=scale,anti_aliasing=anti_aliasing)
    t = plt.imshow(trans)
    plt.colorbar(t)
    plt.show()
    trans = transform.rotate(trans,45,preserve_range=True)

    t = plt.imshow(trans)
    plt.colorbar(t)
    plt.show()

    if trans.shape[0] > orig_shape[0]:
        print(trans.shape)
        new_center_pixel = trans.shape[0]/2 - 0.5
        print(new_center_pixel)
        cutout_range = np.array([new_center_pixel-orig_shape[0]/2,new_center_pixel+orig_shape[0]/2])
        print(cutout_range)
        cutout_range[0] = cutout_range[0] + 0.5
        cutout_range[1] = cutout_range[1] + 0.5
        cutout_range = cutout_range.astype(int)
        print(cutout_range)

        trans = trans[cutout_range[0]:cutout_range[1],cutout_range[0]:cutout_range[1]]
        print(trans.shape)
    elif trans.shape[0] < orig_shape[0]:
        print(trans.shape)
        pad_width = int((orig_shape[0] - trans.shape[0]) / 2)
        new_center_pixel = trans.shape[0]/2 - 0.5
        print(new_center_pixel)
        print(pad_width)
        # Add padding
        trans = np.pad(trans, pad_width, mode='constant', constant_values=0.)
        print(trans.shape)

    t = plt.imshow(trans)
    plt.colorbar(t)
    plt.show()


"""
path_to_data = '/home/collin/projects/distributional_regression/data/'
instrument = 'AIA'
channel = 193
date = datetime(2010,6,1)
series = 'AIA.lev1_euv_12s'

image_file = (path_to_data + instrument
              + '/' + str(channel)
              + '/' + str(date.year)
              + '/' + date.strftime('%m') + '/')
              #+ '/' + series
              #+ '.' + date.strftime('%Y-%m-%d'))
meta_data = pickle.load(open(image_file + 'meta_data.pickle','rb'))
meta_data = meta_data.iloc[0,:]
meta_data = meta_data.drop(meta_data[meta_data.isna()==True].index)
meta_data = list(zip(meta_data.index,meta_data))
header = astropy.io.fits.Header(meta_data)


fits_folder = '/home/collin/projects/distributional_regression/data/AIA/193/2010/06/'
preprocessed_numpy_folder = '/home/collin/projects/distributional_regression/data/preprocessed/aia_193/2010/06/'
#image_file = '/home/collin/projects/distributional_regression/data/AIA/193/2010/06/aia.lev1_euv_12s.2010-06-01T000002Z.193.image.fits'


pointing_table = aiapy.calibrate.util.get_pointing_table(datetime(2010,5,31), datetime(2010,6,2))
psf = pickle.load(open('psf_193.pickle','rb'))
rebin_dimension = [1024,1024]
psf_rebinned = rebin_psf(psf,rebin_dimension)
correction_table = get_correction_table()

files = os.listdir(fits_folder)
files.sort()
for file in files:
    if file[-5:] == '.fits':
        fits_file = fits_folder + file
        product = file[0:3]
        channel = file.split('.image')[0][-3:]
        date_str = file.split('T')[0][-10:]
        time_str = file.split('T')[1][:6]

        date = datetime.strptime(date_str + '_' + time_str,'%Y-%m-%d_%H%M%S')
        print(product,channel,date)

        aia_map = sunpy.map.Map(fits_file)
        preprocessed_aia_map = preprocess_single_aia_image(aia_map,pointing_table,psf_rebinned,correction_table)

        pickle.dump([preprocessed_aia_map,date],open(preprocessed_numpy_folder + date.strftime('%Y-%m-%d_%H:%M') + '_' + channel ,'wb'))

exit()
"""
"""
preprocessed_numpy_folder = '/home/collin/projects/distributional_regression/data/preprocessed/aia_193/2010/06/'
fadil_folder = '/home/collin/projects/distributional_regression/data/fadil/06/'


files = os.listdir(preprocessed_numpy_folder)
files.sort()
for file in files:
    data = pickle.load(open(preprocessed_numpy_folder + file,'rb'))
    img = data[0]
    date = data[1]
    img_fd_sp = sunpy.map.Map(fadil_folder + 'aia_processed_193A_' + date.strftime('%Y_%m_%d') + 'T' + date.strftime('%H_%M') + '.fits')
    img = np.array(img.value)
    img_fd = np.array(img_fd_sp.data)
    dif = np.abs(img - img_fd)
    print(np.max(np.abs(img_fd)),np.max(np.abs(dif)),np.mean(np.abs(img_fd)),np.mean(np.abs(dif)))

    mask = np.logical_and(dif > 5, img_fd > 10)
    dif_rel = np.zeros(dif.shape)
    dif_rel[mask] = np.abs(img[mask] - img_fd[mask]) / np.abs(
        img_fd[mask])
    fig, ax = plt.subplots(2)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(projection=img_fd_sp)
    img_fd_sp.plot(axes=ax1)
    ax[0].imshow(dif, norm='log')
    ax[1].imshow(dif_rel, norm='log')

    image_file = '/home/collin/projects/distributional_regression/data/AIA/193/2010/06/aia.lev1_euv_12s.2010-06-01T000002Z.193.image.fits'
    pointing_table = aiapy.calibrate.util.get_pointing_table(datetime(2010, 5, 31), datetime(2010, 6, 2))
    psf = pickle.load(open('psf_193.pickle', 'rb'))
    rebin_dimension = [1024, 1024]
    psf_rebinned = rebin_psf(psf, rebin_dimension)
    correction_table = get_correction_table()
    aia_map = sunpy.map.Map(image_file)
    preprocessed_aia_map = preprocess_single_aia_image(aia_map, pointing_table, psf_rebinned, correction_table,return_numpy=False)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(projection=preprocessed_aia_map)
    preprocessed_aia_map.plot(axes=ax2)
    plt.show()

exit()
"""

from sunpy.data.sample import AIA_171_IMAGE
from sunpy.map.maputils import all_coordinates_from_map, coordinate_is_on_solar_disk

aia = sunpy.map.Map(AIA_171_IMAGE)
print(int(aia.dimensions[0].value) == 1024)
#psf = aiapy.psf.psf(193*u.Angstrom,use_gpu=False)
#pickle.dump(psf,open('psf_193.pickle','wb'))
psf = pickle.load(open('psf_211.pickle', 'rb'))
print(psf.shape)
exit()
hpc_coords = all_coordinates_from_map(aia)
mask = coordinate_is_on_solar_disk(hpc_coords)
palette = aia.cmap.copy()
palette.set_bad('black')
scaled_map = sunpy.map.Map(aia.data, aia.meta, mask=mask)
fig = plt.figure()
ax = fig.add_subplot(projection=scaled_map)
scaled_map.plot(axes=ax, cmap=palette)
scaled_map.draw_limb(axes=ax)



image_file = '/home/collin/projects/distributional_regression/data/AIA/193/2010/06/aia.lev1_euv_12s.2010-06-01T000002Z.193.image.fits'
aia = sunpy.map.Map(image_file)
hpc_coords = all_coordinates_from_map(aia)
mask = coordinate_is_on_solar_disk(hpc_coords)
palette = aia.cmap.copy()
palette.set_bad('black')
scaled_map = sunpy.map.Map(aia.data, aia.meta, mask=mask)
fig2 = plt.figure()
ax2 = fig2.add_subplot(projection=scaled_map)
scaled_map.plot(axes=ax2, cmap=palette)
scaled_map.draw_limb(axes=ax2)
plt.show()



print(sunpy.map.is_all_on_disk(aia_map))
sunpy.map.coordinate_is_on_solar_disk(coordinates=...)

fig = plt.figure()
ax = fig.add_subplot(projection=aia_map)
aia_map.plot(axes=ax)
aia_map.draw_limb(axes=ax)

print(aia_map.rsun_obs,aia_map.meta["r_sun"], aia_map.meta["cdelt1"],aia_map.scale[0],aia_map.rsun_obs.value/aia_map.scale[0].value)
print(aia_map.reference_pixel)
circle = plt.Circle((aia_map.reference_pixel[0].value,aia_map.reference_pixel[1].value), aia_map.rsun_obs.value/aia_map.scale[0].value, color='red', fill=False)
#circle = plt.Circle((aia_map.reference_pixel[0].value,aia_map.reference_pixel[1].value), 960.0/aia_map.scale[0].value, color='red', fill=False)

#ax.add_patch(circle)

plt.show()
exit()


image_file = '/home/collin/projects/distributional_regression/data/AIA/193/2010/06/aia.lev1_euv_12s.2010-06-01T000002Z.193.image.fits'
aia_map = sunpy.map.Map(image_file)

pointing_table = aiapy.calibrate.util.get_pointing_table(datetime(2010, 5, 31), datetime(2010, 6, 2))
psf = pickle.load(open('psf_193.pickle', 'rb'))
rebin_dimension = [1024, 1024]
psf_rebinned = rebin_psf(psf, rebin_dimension)
correction_table = get_correction_table()
aia_map = sunpy.map.Map(image_file)
print(aia_map.wcs)
preprocessed_aia_map = preprocess_single_aia_image(aia_map, pointing_table, psf_rebinned, correction_table,return_numpy=False)
print(preprocessed_aia_map.wcs)
exit()

in_time = preprocessed_aia_map.date
out_time = in_time + 24*u.hour
out_frame = Helioprojective(observer='earth', obstime=out_time,
                            rsun=preprocessed_aia_map.coordinate_frame.rsun)
out_center = SkyCoord(0*u.arcsec, 0*u.arcsec, frame=out_frame)
header = sunpy.map.make_fitswcs_header(preprocessed_aia_map.data.shape,
                                       out_center,
                                       scale=u.Quantity(preprocessed_aia_map.scale))
out_wcs = WCS(header)
with propagate_with_solar_surface():
    out_warp = preprocessed_aia_map.reproject_to(out_wcs)
print(out_warp.wcs)

fig = plt.figure(figsize=(12, 4))

ax1 = fig.add_subplot(121, projection=preprocessed_aia_map)
preprocessed_aia_map.plot(axes=ax1, vmin=0, vmax=20000, title='Original map')
plt.colorbar()

ax2 = fig.add_subplot(122, projection=out_warp)
out_warp.plot(axes=ax2, vmin=0, vmax=20000,
              title=f"Reprojected to an Earth observer {(out_time - in_time).to('day')} later")
plt.colorbar()

plt.show()
exit()

image_file = '/home/collin/projects/distributional_regression/data/HMI/2010/06/hmi.m_720s.20100601_000000_TAI.1.magnetogram.fits'
hmi_map = sunpy.map.Map(image_file)

hmi_map = hmi_map.resample([1024,1024]*u.pixel)

hmi_reprojected = hmi_map.reproject_to(preprocessed_aia_map.wcs)

exit()

"""
fig = plt.figure()
ax1 = fig.add_subplot(projection=aia_map)
aia_map.plot(axes=ax1, clip_interval=(1, 99.9)*u.percent)
plt.show()



#preprocessed_aia_map = preprocess_single_aia_image(aia_map, pointing_table, psf_rebinned, correction_table,return_numpy=False)



image_file = '/home/collin/projects/distributional_regression/data/HMI/2010/06/hmi.m_720s.20100601_000000_TAI.1.magnetogram.fits'
#pointing_table = aiapy.calibrate.util.get_pointing_table(datetime(2010, 5, 31), datetime(2010, 6, 2))
#psf = pickle.load(open('psf_193.pickle', 'rb'))
#rebin_dimension = [1024, 1024]
#psf_rebinned = rebin_psf(psf, rebin_dimension)
#correction_table = get_correction_table()
hmi_map = sunpy.map.Map(image_file).resample([1024,1024]*u.pixel)
print(hmi_map.reference_pixel,hmi_map.rsun_obs,hmi_map.scale,hmi_map.rotation_matrix)
print(hmi_map.data.shape)
print(hmi_map.meta['QUALITY'])



fig2 = plt.figure()
hmi_map.plot_settings['cmap'] = "hmimag"
hmi_map.plot_settings['norm'] = plt.Normalize(-1500, 1500)
ax2 = fig2.add_subplot(projection=hmi_map)
hmi_map.plot(axes=ax2)
plt.show()


hmi_reprojected = hmi_map.reproject_to(aia_map.wcs)
print(hmi_map.reference_pixel,hmi_map.rsun_obs,hmi_map.scale,hmi_map.rotation_matrix)





#preprocessed_aia_map = preprocess_single_aia_image(aia_map, pointing_table, psf_rebinned, correction_table,return_numpy=False)
fig2 = plt.figure()
hmi_reprojected.plot_settings['cmap'] = "hmimag"
hmi_reprojected.plot_settings['norm'] = plt.Normalize(-1500, 1500)
ax2 = fig2.add_subplot(projection=hmi_reprojected)
hmi_reprojected.plot(axes=ax2)
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(projection=aia_map)
aia_map.plot(axes=ax1, clip_interval=(1, 99.9)*u.percent)
hmi_reprojected.plot(axes=ax1, alpha=0.5)
plt.title('HMI overlaid on AIA')

plt.show()

exit()
"""


image_file = '/home/collin/projects/distributional_regression/data/AIA/193/2010/06/aia.lev1_euv_12s.2010-06-01T000002Z.193.image.fits'

image_data = fits.getdata(image_file)
#print(image_data)
#print(image_data.shape)
aia_map = sunpy.map.Map(image_file)

#pointing_table = aiapy.calibrate.util.get_pointing_table(datetime(2010,5,31), datetime(2010,6,2))
#psf = pickle.load(open('psf_193.pickle','rb'))
#rebin_dimension = [1024,1024]
#psf_rebinned = rebin_psf(psf,rebin_dimension)
#correction_table = get_correction_table()

#print('start preprocessing')
#aia_map_preprocessed = preprocess_single_aia_image(aia_map,pointing_table,psf_rebinned,correction_table)
#print('finished preprocessing')

from sunpy.data.sample import AIA_171_IMAGE
aia = sunpy.map.Map(AIA_171_IMAGE)
aia = sunpy.map.Map(AIA_171_IMAGE)
mask = coordinate_is_on_solar_disk(hpc_coords)
palette = aia.cmap.copy()
palette.set_bad('black')
scaled_map = sunpy.map.Map(aia.data, aia.meta, mask=mask)
fig = plt.figure()
ax = fig.add_subplot(projection=scaled_map)
scaled_map.plot(axes=ax, cmap=palette)
scaled_map.draw_limb(axes=ax)
plt.show()

print(sunpy.map.is_all_on_disk(aia_map))
sunpy.map.coordinate_is_on_solar_disk(coordinates=...)

fig = plt.figure()
ax = fig.add_subplot(projection=aia_map)
aia_map.plot(axes=ax)
aia_map.draw_limb(axes=ax)

print(aia_map.rsun_obs,aia_map.meta["r_sun"], aia_map.meta["cdelt1"],aia_map.scale[0],aia_map.rsun_obs.value/aia_map.scale[0].value)
print(aia_map.reference_pixel)
circle = plt.Circle((aia_map.reference_pixel[0].value,aia_map.reference_pixel[1].value), aia_map.rsun_obs.value/aia_map.scale[0].value, color='red', fill=False)
#circle = plt.Circle((aia_map.reference_pixel[0].value,aia_map.reference_pixel[1].value), 960.0/aia_map.scale[0].value, color='red', fill=False)

#ax.add_patch(circle)

plt.show()



'''
dif = np.abs(aia_map_registered_small.data - aia_map_own_registered_small.data)
print(np.mean(dif),np.max(dif))
mask = np.logical_and(dif > 5, aia_map_registered_small.data > 10)
dif_rel = np.zeros(dif.shape)
dif_rel[mask] = np.abs(aia_map_registered_small.data[mask] - aia_map_own_registered_small.data[mask])/np.abs(aia_map_registered_small.data[mask])
print(np.mean(dif_rel),np.max(dif_rel))
fig,ax = plt.subplots(2)
fig1 = plt.figure()
fig2 = plt.figure()
ax1 = fig1.add_subplot(projection=aia_map_registered_small)
aia_map_registered_small.plot(axes=ax1)
ax2 = fig2.add_subplot(projection=aia_map_own_registered_small)
aia_map_own_registered_small.plot(axes=ax2)
ax[0].imshow(dif,norm='log')
ax[1].imshow(dif_rel,norm='log')
'''

#fig,ax = plt.subplots()

#ax.imshow(psf,norm=ImageNormalize(vmin=1e-8, vmax=1e-3, stretch=LogStretch()),origin="lower",)
#psf = aiapy.psf.psf(aia_map.wavelength)
#pickle.dump(psf,open('psf_193.pickle','wb'))



# radius of sun always the same?

exit()