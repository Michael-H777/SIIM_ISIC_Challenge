from sys import platform

# data processing configs 
test_size = 20
train_size = 1500
force_source = True
augment_times = 4

patch_size = 256
stride = 100

input_file_name = 'CO2_Fast_P_low.tif'
target_file_name = 'CO2_HQ_P_Low.tif'

if 'linux' in platform.lower():
    raw_data_folder = '/home/michael/ssd_cache/SMART/raw_data'
    processed_data_folder = '/home/michael/ssd_cache/SMART/processed_data'
else:
    raw_data_folder = '//192.168.1.4/SSD_Cache/SMART/raw_data'
    processed_data_folder = '//192.168.1.4/SSD_Cache/SMART/processed_data'

# training configs 
max_epochs = 100
batch_size = 16
initial_learning_rate = 1e-3
epoch_samples = 5_000
updates = 100

# hardware configs
gpu_id = [0]

# reporting config 
line_width = 85
