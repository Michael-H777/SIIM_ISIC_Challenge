from sys import platform

# data processing configs 
test_size = 20
train_size = 800
force_source = True
augment_times = 2

patch_size = 256
stride = 100

input_file_name = 'CO2_Fast_P_low.tif'
target_file_name = 'CO2_HQ_P_Low.tif'
#input_file_name = 'CO2_HQ_P_Low.tif'

if 'linux' in platform.lower():
    raw_data_folder = '/home/michael/ssd_cache/SMART/raw_data'
    processed_data_folder = '/home/michael/ssd_cache/SMART/processed_data'
else:
    raw_data_folder = 'C:/Users/Michael/Documents/SMART/raw_data'
    processed_data_folder = 'C:/Users/Michael/Documents/SMART/processed_data'

# reporting config 
line_width = 85
