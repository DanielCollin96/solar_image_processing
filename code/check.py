from datetime import datetime,timedelta
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
from utils import (read_file_name, check_file_quality, find_missing_preprocessed_dates, load_existing_raw_files,
                   load_config_data, find_files_to_preprocess, check_completeness_of_preprocessed_images,
                   find_missing_cropped_dates, load_existing_preprocessed_dates,
                   create_folders_for_preprocessed_images)
from astropy.time import Time
from pathlib import Path
from start_preprocessing import SolarImagePreprocessor
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
from start_preprocessing import AIAPreprocessor

cropped = np.load('/home/collin/projects/distributional_regression/solar_images/data/preprocessed_images/dl_cropped/aia_193/2015/01/193_2015-01-01_23:00.npy')
fig, ax = plt.subplots()
ax.imshow(cropped)

cropped_old = np.load('/home/collin/projects/distributional_regression/helmholtz_ai/data/solar_images/aia_193/2015/01/193_20150101_2300.npy')
fig, ax = plt.subplots()
ax.imshow(cropped_old)

fig,ax = plt.subplots()
ax.imshow((cropped_old - cropped)/cropped_old)
print(np.max((cropped_old - cropped)/(cropped_old+1e-8)))

plt.show()

raw = sunpy.map.Map(
    '/home/collin/projects/distributional_regression/solar_images/data/unprocessed_images/SDO/AIA/171/2015/01/aia_lev1_171a_2015_01_01t00_00_11_34z_image_lev1.fits')
fig, ax = plt.subplots()
ax.imshow(raw.data)

preprocessed = np.load('/home/collin/projects/distributional_regression/solar_images/data/preprocessed_images/deep_learning/aia_171/2015/01/171_2015-01-01_00:00.npy')
fig, ax = plt.subplots()
ax.imshow(preprocessed)

raw_next_hour = sunpy.map.Map(
    '/home/collin/projects/distributional_regression/solar_images/data/unprocessed_images/SDO/AIA/171/2015/01/aia_lev1_171a_2015_01_01t23_00_11_34z_image_lev1.fits')
fig, ax = plt.subplots()
ax.imshow(raw_next_hour.data)

preprocessed_next_hour = np.load('/home/collin/projects/distributional_regression/solar_images/data/preprocessed_images/deep_learning/aia_171/2015/01/171_2015-01-01_23:00.npy')
fig, ax = plt.subplots()
ax.imshow(preprocessed_next_hour)

resolution = raw.data.shape[0]
if resolution == 4096:
    aia_map = raw.resample([1024, 1024] * u.pixel)

base_path = Path(os.getcwd()).parent
config_path = base_path / 'data' / 'configuration_data'
psf, correction_table, pointing_table = load_config_data(config_path, '171', datetime(2015,1,1))
print('Processing')
aia_preprocessor = AIAPreprocessor(pointing_table, psf,correction_table)
rotated, meta_info = aia_preprocessor.preprocess_image(aia_map,datetime(2015,1,1,0),datetime(2015,1,1,23))
fig, ax = plt.subplots()
ax.imshow(rotated)

plt.show()