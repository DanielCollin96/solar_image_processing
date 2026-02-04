import os
from datetime import datetime
from dateutil.relativedelta import relativedelta


path_to_files = '../../../topological_data_analysis/data/solar_images/full_disk_cropped/aia_193/'

start = datetime(2010,5,1)
end = datetime(2024,6,30)

current_month = start

while current_month < end:
    path_to_monthly_files = path_to_files + current_month.strftime('%Y/%m')
    files = os.listdir(path_to_monthly_files)
    files.sort()
    for file in files:
        date = datetime.strptime(file[4:].split('.npy')[0],'%Y-%m-%d_%H:%M')
        print('Replacing',path_to_monthly_files + '/' + file,'by',path_to_monthly_files + '/' + file[:4] + date.strftime('%Y%m%d_%H%M') + '.npy')
        os.rename(path_to_monthly_files + '/' + file,path_to_monthly_files + '/' + file[:4] + date.strftime('%Y%m%d_%H%M') + '.npy')
    current_month = current_month + relativedelta(months=1)