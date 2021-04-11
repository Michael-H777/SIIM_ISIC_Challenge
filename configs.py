from sys import platform

# data processing configs 

positive_augmnets = 6

patch_size = 256

if 'linux' in platform.lower():
    raw_data_folder = '/home/michael/ssd_cache/Lesion/raw_data'
    processed_data_folder = '/home/michael/ssd_cache/Lesion/processed_data'
else:
    raw_data_folder = 'D:/Data/Lesion/raw_data'
    processed_data_folder = 'C:/Users/Michael/Documents/Data/Lesion/processed_data'

# reporting config 
line_width = 90
