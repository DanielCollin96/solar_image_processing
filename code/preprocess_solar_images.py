from start_preprocessing import SolarImagePreprocessor, ImageCropper
from datetime import datetime,timedelta
import os
from pathlib import Path



# update readme files in solardl

def main():
    base_path = Path(os.getcwd()).parent

    channels_to_process = ['aia_171','aia_193','aia_211','hmi']
    start = datetime(2010,5,1)
    end = datetime(2024,6,30,23) #datetime.today() #

    paths = {'raw': base_path / 'data' / 'unprocessed_images' / 'SDO',
             'preprocessed': base_path / 'data' / 'preprocessed_images' / 'deep_learning',
             'cropped': base_path / 'data' / 'preprocessed_images' / 'full_disk_cropped',#'dl_cropped',
             'config': base_path / 'data' / 'configuration_data'}

    #preprocessor = SolarImagePreprocessor(start,end,channels_to_process,paths)
    #preprocessor.run(load_preprocessing_fails=False,overwrite_existing=False)

    cropper = ImageCropper(start,end,channels_to_process,paths,downsample_resolution=256,crop_square_in_downsampled='disk',resize_cropped=None)
    cropper.run()


if __name__ == "__main__":
    print('Starting script')
    main()
    print('Finished script')
