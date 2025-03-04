from datetime import datetime

def read_file_name(file,preprocessed=False):
    if preprocessed:
        if file[0:3] == '193' or file[0:3] == '211':
            product = 'aia'
            channel = file[0:3]
        elif file[0:3] == 'hmi':
            product = 'hmi'
            channel = ''
        date_str = file.split('.pickle')[0][4:]
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
