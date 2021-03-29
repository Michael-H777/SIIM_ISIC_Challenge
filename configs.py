from sys import platform

# data processing configs 

patch_size = 256
stride = 100


if 'linux' in platform.lower():
    raw_data_folder = '/home/michael/ssd_cache/lesion/raw_data'
    processed_data_folder = '/home/michael/ssd_cache/lesion/processed_data'
else:
    raw_data_folder = 'C:/Users/Michael/Documents/lesion/raw_data'
    processed_data_folder = 'C:/Users/Michael/Documents/lesion/processed_data'

# reporting config 
line_width = 85
