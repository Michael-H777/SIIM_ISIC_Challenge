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


class dummy_semi_loader:
    
    def __init__(self, *, options, supervised, unsupervised):
        self.epoch_batches = options.epoch_updates
        self.batches = int(options.epoch_samples/options.batch_size/options.epoch_updates)
        self.supervised = supervised
        self.unsupervised = unsupervised
        
    def __len__(self):
        return len(self.unsupervised)
    
    def __iter__(self):
        for _ in range(self.epoch_batches):
            if random.randint(0, 5):
                for _ in range(self.batches):
                    yield next(self.unsupervised.iterator)
            else:
                for _ in range(self.batches):
                    yield next(self.supervised.iterator)


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
    for augment_type in methods + random.sample(range(1, 14), augment_times):
        
        processed_data = np.zeros((len(patches), *patches[0].shape), dtype=np.float32)
        # do same augment on all index 
        for image_index, image in enumerate(patches):
            processed_data[image_index,:,:] = augment_data(image.copy(), augment_type)
            
        h5file.create_dataset(f'{name_header}_{augment_type}', data=processed_data.copy())
        
    return h5file


def process_table(table, h5_file, data_scaler, augment_times, force_source): 
    total_size = table.shape[0] 
    for index, series in table.iterrows(): 
        index += 1
        series = series.to_dict()
        image_name = series['image_name']
        target = series['target']
        print(f'\r[{index}/{total_size}], {image_name=}', end='', flush=True)
        
        # do the things
        image = pydicom.dcmread(f'{raw_data_folder}/2020_train/{image_name}.dcm')
        image = cv2.resize(image.pixel_array, (patch_size, patch_size), interpolation=cv2.INTER_LANCZOS4)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # fill package
        blank = np.zeros((5, patch_size, patch_size))
        blank[0] = image[:,:,0]
        blank[1] = image[:,:,1]
        blank[2] = image[:,:,2]
        blank[3] = gray 
        
        input_data = [data_scaler.to_target(blank[index].copy()) for index in range(4)]
        input_data += [np.ones((patch_size, patch_size)) * target]
        
        if target:
            create_h5_dataset(input_data, image_name, h5_file, augment_times, force_source)
        else:
            create_h5_dataset(input_data, image_name, h5_file, 1, False)
            
    print()
    h5_file.close() 
    return None 


def make_model_data():
    
    data_scaler = DataScaler(domain_min=0, 
                             domain_max=255, 
                             target_min=0, target_max=1)
    
    images = pd.read_csv(f'{raw_data_folder}/2020_label.csv')
    
    positive = images.loc[images['target']==1].sample(frac=1)
    negative = images.loc[images['target']==0].sample(frac=1)
    
    positive_train, positive_test = positive.iloc[:-test_size], positive.iloc[-test_size:]
    negative_train, negative_test = negative.iloc[:-test_size*10], negative.iloc[-test_size*10:]
    
    train_table = pd.concat([positive_train, negative_train]).reset_index(drop=True)
    test_table = pd.concat([positive_test, negative_test]).reset_index(drop=True)
    
    labeled_h5 = h5py.File(f'{processed_data_folder}/labeled_train.h5', 'w')
    process_table(train_table, labeled_h5, data_scaler, positive_augmnets, True)
    
    test_h5 = h5py.File(f'{processed_data_folder}/test_data.h5', 'w')
    process_table(test_table, test_h5, data_scaler, 2, True)
    
    print('data processing complete')
    return None 
    

def augment_data(image, mode):
    # all rotation is counter clock wise 
    if mode == 0:
        out = image
    elif mode == 1:
        # flip up and down
        out = np.flipud(image)
    elif mode == 2:
        # flip up and down
        out = np.flipud(image)
        out = np.transpose(out)
        
    elif mode == 3:
        # rotate 90 degree
        out = np.rot90(image)
    elif mode == 4:
        # rotate 90 degree
        out = np.rot90(image)
        out = np.transpose(out)
        
    elif mode == 5:
        # rotate 90 degree and flip up and down
        out = np.rot90(image)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 90 degree and flip up and down
        out = np.rot90(image)
        out = np.flipud(out)
        out = np.transpose(out)
        
    elif mode == 7:
        # rotate 180 degree
        out = np.rot90(image, k=2)
    elif mode == 8:
        # rotate 180 degree
        out = np.rot90(image, k=2)
        out = np.transpose(out)
        
    elif mode == 9:
        # rotate 180 degree and flip
        out = np.rot90(image, k=2)
        out = np.flipud(out)
    elif mode == 10:
        # rotate 180 degree and flip
        out = np.rot90(image, k=2)
        out = np.flipud(out)
        out = np.transpose(out)
        
    elif mode == 11:
        # rotate 270 degree
        out = np.rot90(image, k=3)
    elif mode == 12:
        # rotate 270 degree
        out = np.rot90(image, k=3)
        out = np.transpose(out)
        
    elif mode == 13:
        # rotate 270 degree and flip
        out = np.rot90(image, k=3)
        out = np.flipud(out)
    elif mode == 14:
        # rotate 270 degree and flip
        out = np.rot90(image, k=3)
        out = np.flipud(out)
        out = np.transpose(out)
    return out


if __name__ == '__main__':
    make_model_data() 
    