from configs import * 
from packages import * 


class _RepeatSampler(object):
    
    # https://github.com/pytorch/pytorch/issues/15849

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class FastDataLoader(torch.utils.data.dataloader.DataLoader):
    
    # https://github.com/pytorch/pytorch/issues/15849

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class DataSet(torch.utils.data.Dataset): 
    
    def __init__(self, *, source_file_name):
        self.filename = source_file_name 
        with h5py.File(f'{processed_data_folder}/{self.filename}', 'r') as filein: 
            self.names = list(filein.keys())
        random.shuffle(self.names)
            
    def __len__(self):
        return len(self.names)
        
    def __getitem__(self, index):
        self.name = self.names[index]
        with h5py.File(f'{processed_data_folder}/{self.filename}', 'r') as filein: 
            data = np.float32(filein[self.name])
        return torch.from_numpy(data)


class DataScaler:
    
    def __init__(self, *, domain_min, domain_max, target_min=0, target_max=1):
        self.domain_min = domain_min
        self.domain_range = domain_max - domain_min
        
        self.target_min = target_min
        self.target_range = target_max - target_min
        
    def get_domain_stats(self, folder_path):
        domain_min, domain_max = np.inf, -np.inf
        for filename in os.listdir(folder_path):
            data = tifffile.imread(f'{folder_path}/{filename}')
            domain_min = np.min(domain_min, data.min())
            domain_max = np.max(domain_max, data.max())
        # update stats
        self.domain_min = domain_min
        self.domain_range = domain_max - domain_min
        return None 

    def to_target(self, input_data):
        position = (input_data - self.domain_min) / (self.domain_range)
        result = (self.target_range * position) + self.target_min
        return np.float32(result)
    
    def to_domain(self, input_data): 
        position = (input_data - self.target_min) / self.target_range
        result = position * self.domain_range + self.domain_min
        return np.float32(result)


def create_h5_dataset(patches, name_header, h5file, augment_times=0, force_source=True):
    
    assert isinstance(patches, list)
    
    methods = [0] if force_source else []

    # if force_source, will guarantee original patch in h5file
    for augment_type in methods + random.sample(range(1, 8), augment_times):
        
        processed_data = np.zeros((len(patches), *patches[0].shape))
        # do same augment on all index 
        for image_index, image in enumerate(patches):
            processed_data[image_index,:,:] = augment_data(image.copy(), augment_type)
            
        h5file.create_dataset(f'{name_header}_{augment_type}', data=processed_data.copy())
        
    return h5file


def make_model_data(): 
    global train_size, test_size, augment_times, force_source 
    global raw_data_folder, processed_data_folder
    
    print('loading input data')
    input_files = [tifffile.imread(f'{raw_data_folder}/{input_file_name}')]
    print('loading target data')
    target_file = tifffile.imread(f'{raw_data_folder}/{target_file_name}')

    domain_min = min([np.ma.masked_equal(array, 0.0, copy=False).min() for array in input_files + [target_file]])
    domain_max = max([np.ma.masked_equal(array, 0.0, copy=False).max() for array in input_files + [target_file]])

    data_scaler = DataScaler(domain_min=domain_min, 
                             domain_max=domain_max, 
                             target_min=0, target_max=1)
    
    with open(f'{processed_data_folder}/data_scaler.pickle', 'wb') as fileout:
        pickle.dump(data_scaler, fileout)
    
    all_index = list(range(250, 1800))
    random.shuffle(all_index)
    train_index, test_index = all_index[:train_size], all_index[-test_size:]
    
    labeled_h5 = h5py.File(f'{processed_data_folder}/labeled_train.h5', 'w')
    
    for progress, image_index in enumerate(train_index, 1):
        # have this report for each image slice processed 
        used_patch = dropped_patch = 0 
        for row_num, row_index in enumerate(range(0, input_files[0].shape[1] - patch_size, stride)):
            
            for col_num, col_index in enumerate(range(0, input_files[0].shape[2] - patch_size, stride)):
                
                input_patch = [array[image_index , row_index:row_index+patch_size , col_index:col_index+patch_size].copy() for array in input_files]
                
                if np.any([array == 0 for array in input_patch]):
                    dropped_patch += force_source + augment_times
                    continue 
                # now store them 
                patches = input_patch + [target_file[image_index , row_index:row_index+patch_size , col_index:col_index+patch_size].copy()]
                patches = [data_scaler.to_target(array) for array in patches]
                
                create_h5_dataset(patches=patches, name_header=f'{image_index}_{row_num}_{col_num}', h5file=labeled_h5, 
                                  augment_times=augment_times, force_source=force_source)
                used_patch += force_source + augment_times
            # exited col_num for loop
        # exited row_num for loop 
        print(f'\r[{progress:>{len(str(train_size))}}/{train_size}] processed {image_index:>4}, valid patches {used_patch}, dropped {dropped_patch}', end='', flush=True)
    # exited image_index for loop 
    labeled_h5.close()
    
    ################################################################
    # implement semi-supervised learning data processing 
    ################################################################
    
    test_h5 = h5py.File(f'{processed_data_folder}/test_data.h5', 'w')
    for image_index in test_index: 
        
        input_patch = [array[image_index , 144:656 , 144:656].copy() for array in input_files]
        patches = input_patch + [target_file[image_index , 144:656 , 144:656].copy()]
        patches = [data_scaler.to_target(array) for array in patches]
        
        create_h5_dataset(patches=patches, name_header=f'{image_index}', h5file=test_h5)
    test_h5.close() 
    
    print('\ndata processing complete')
    
    return None


def augment_data(image, mode):
    # all rotation is counter clock wise 
    if mode == 0:
        out = image
    elif mode == 1:
        # flip up and down
        out = np.flipud(image)
    elif mode == 2:
        # rotate 90 degree
        out = np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(image)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(image, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(image, k=3)
        out = np.flipud(out)
    return np.transpose(out)
