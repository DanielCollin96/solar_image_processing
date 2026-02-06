import os
from datetime import datetime, timedelta
import pickle
import pandas as pd
from dateutil.relativedelta import relativedelta
from joblib import Parallel, delayed, cpu_count
import numpy as np

def collect_shapes(file,path):
    if file[:3] == '193':
        print('Reading', file)
        data = pickle.load(open(path+file,'rb'))
        arr = data[0]
        date = data[1]
        shape = arr.shape[0]

        return [date,shape]
    else:
        return None


wl = 193
read_maps = False
read_meta = True

dir_path = os.path.dirname(os.getcwd())


start = datetime(2010,5,1)
end = datetime(2024,6,30,23)
path_to_aia = dir_path + '/SDO/AIA'
path_to_hmi = dir_path + '/SDO/HMI/magnetogram'
path_to_preprocessed = dir_path + '/images_preprocessed'

path_to_output = path_to_preprocessed + '/aia_{}'.format(str(wl))

current_month = datetime(start.year, start.month, 1)
meta_list = []
map_list = []

while current_month <= end:

    month_str = current_month.strftime('%Y/%m')

    path_to_month = path_to_output + '/' + month_str + '/'

    if read_maps:
        files = os.listdir(path_to_month)
        files.sort()

        collect = []

        res = Parallel(n_jobs=-1)(
            delayed(collect_shapes)(file, path_to_month) for file in files)
        for item in res:
            if item is not None:
                map_list.append(item)
    if read_meta:
        try:
            print('Reading', month_str)
            meta = pickle.load(open(path_to_month + 'meta.pickle', 'rb'))
            shape = []
            for wcs in meta[:,2]:
                shape.append(wcs.array_shape[0])

            meta = pd.DataFrame(np.stack([meta[:, 1],np.array(shape)],axis=1), index=meta[:, 0])
            meta_list.append(meta)
        except:
            print(month_str, 'not found')


    current_month = current_month + relativedelta(months=1)


if read_maps:
    pickle.dump(np.array(map_list), open('map_list.pickle', 'wb'))

if read_meta:
    meta_list = pd.concat(meta_list, axis=0)
    meta_list = meta_list.sort_index()
    pickle.dump(meta_list, open('meta.pickle', 'wb'))

print('FINISHED SUCCESSFULLY')
