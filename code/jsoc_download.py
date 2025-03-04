import datetime as dt
import os
import tarfile
import drms
import numpy as np

"""
Download client for JSOC (i.e., AIA, HMI, and MDI data).

Usage:
    1) create a client object:
        client = client(email address)  

            #The email must be registered at http://jsoc.stanford.edu/ajax/exportdata.html .
            #If this has not been done before, go to http://jsoc.stanford.edu/ajax/exportdata.html and put your email address into the Notify field . It will then ask you if you want to register the email address and gives further instructions.

    2) create a request string:
        rs = client.create_request_string(series, starttime, endtime = '', wavelength = '', segment = '', period = '', cadence = '').

            series:               Series name. Can be found at the JSOC Lookdata. E.g., 'AIA.lev1_euv_12s' for AIA EUV images, or 'hmi.M_45s' for HMI line-of-sight magnetograms.
            starttime,  endtime:  Either date in format YYYY-MM-DDThh:mm:ss, or a datetime object
            period:               Can be used instead of endtime. Either a string as 'XdXhXmXs' or as timedelta object.
            cadence:              The cadence requested. Either as string as 'XdXhXmXs' or as timedelta object.
            wavelength:           For AIA images, the wavelength of the filter.
            segment:              The data product. Usually 'image' or 'magnetogram'.

    3) Optional: Look if the request string is valid, and/or request additional fits keywords to filter the search request in the next step
        search_results = client.search(rs, keys = keys)

            #keys: Optional. Array of additional fits keywords to be requested. E.g., ['t_obs','EXPTIME']

    4) Download the data
        files_downloaded = client.download(rs, download_dir, method = 'url-tar', protocol = 'fits', filter = None, rebin = 1, process = {})

            download_dir:  Target download directory
            method:        Download method. Default is 'url-tar' for bulk requests and 'url' if filter is used.
            protocol:      Image type. Default is 'fits'
            filter:        Can be used to download only files from client.search() which match given properties. E.g., search_results['EXPTIME'] < 7 #seconds.
            rebin:         Request to rebin the files on the server before the downloading. E.g., rebin=2 results in images having half the original edge lengths.
            process:       Can be used for further preprocessing of the images besides rebinning. To find possible arguements, go to http://jsoc.stanford.edu/ajax/exportdata.html , check the box 'Enable Processing', set the requested preprocessing, check the box 'check to show export params.', and extract the preprocessing parameters from the params string. E.g., process = {'rebin': {'u':1,'scale':1./rebin,'method':'boxcar'} }

    Limitation:
        JSOC only allows reqests smaller than 100 GB and shorter than 10000 results. For large requests, you have to split your download request in smaller chunks.

    Written by Stefan Hofmeister. 
    Please report bugs to stefan.hofmeister@columbia.edu .
"""


class client:
    _client = None

    def __init__(self, email, verbose=False):
        self._create_client(email, verbose)

    def _create_client(self, email, verbose):
        self._client = drms.Client(email=email)  # , verbose = verbose)

    def create_request_string(self, series, starttime, endtime='', wavelength='', segment='', period='', cadence=''):
        request_string = series
        if endtime:
            if not isinstance(starttime, dt.datetime): starttime = dt.datetime.strptime(starttime, '%Y-%m-%dT%H:%M:%S')
            if not isinstance(endtime, dt.datetime): endtime = dt.datetime.strptime(endtime, '%Y-%m-%dT%H:%M:%S')
            period = abs(starttime - endtime)
        if period:
            if isinstance(period, dt.timedelta): period = str(period.total_seconds()) + 's'
            period = '/' + period
        if cadence:
            if isinstance(cadence, dt.timedelta): cadence = str(cadence.total_seconds()) + 's'
            cadence = '@' + cadence
        request_string += '[{:}{:}{:}]'.format(starttime, period, cadence)

        if wavelength:
            wavelength = '[{:}]'.format(wavelength)
            request_string += wavelength
        if segment:
            segment = '{{{:}}}'.format(segment)
            request_string += segment
        return request_string

    def search(self, request_string, keys=['t_obs', 'EXPTIME']):
        keys = ','.join(keys)
        keys = self._client.query(request_string, key=keys)
        return keys

    def download(self, request_string, download_dir, method='url-tar', protocol='fits', filter=None, rebin=1,
                 process={}):
        download_dir += '/'
        if not os.path.isdir(download_dir): os.makedirs(download_dir)

        if filter:
            filter = np.array(filter)
            method = 'url'

        if rebin != 1:
            process['rebin'] = {'u': 1, 'scale': 1. / rebin, 'method': 'boxcar'}

        export_request = self._client.export(request_string, method=method, protocol=protocol, process=process)

        if 'tar' in method:
            tar_downloaded = export_request.download(download_dir)['download'][0]
            with tarfile.open(tar_downloaded, "r") as tf:
                members, names = tf.getmembers(), tf.getnames()
                valid_fits = ['.fits' in filename for filename in names]
                files_downloaded = [os.path.basename(name) for name, valid in zip(names, valid_fits) if valid == True]
                members_to_extract = [member for member, valid in zip(members, valid_fits) if valid == True]
                tf.extractall(path=download_dir, members=members_to_extract)
                os.remove(tar_downloaded)
        if not 'tar' in method:
            if np.any(filter):
                index = np.where(filter == True)[0]
                files_downloaded = export_request.download(download_dir, index=index)['download'][index]
            else:
                files_downloaded = export_request.download(download_dir)['download']
            files_downloaded = [os.path.basename(file) for file in files_downloaded]
            for i, file in enumerate(files_downloaded):
                name, ending = os.path.splitext(file)
                if ending != '.fits':
                    os.replace(download_dir + file, download_dir + name)
                    files_downloaded[i] = name
        return files_downloaded





