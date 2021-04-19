import data 
from packages import * 
from configs import * 

from loss import * 
from models.networks import * 

parser = argparse.ArgumentParser(description='train file')
parser.add_argument('--preprocess', type=bool, default=False, help='process training data using specified directory')
parser.add_argument('--max_lr', type=float, default=1e-2, help='initial learning rate, decay to end_learning_rate with cosine annihilation')
parser.add_argument('--min_lr', type=float, default=1e-3, help='ending learning rate')
parser.add_argument('--max_epoch', type=int, default=50, help='maximum epoch for training')
parser.add_argument('--epoch_samples', type=int, default=5_000, help='samples for each epochs')
parser.add_argument('--epoch_updates', type=int, default=100, help='allowed updates for each epochs')
parser.add_argument('--batch_size', type=int, default=4, help='batch size when training')
parser.add_argument('--fine_tune_last', type=bool, default=False, help='fine tune last model?')
parser.add_argument('--model_pickle_path', type=str, default='', help='pickled model path, use with scheduler')
parser.add_argument('--log_path', type=str, default='/home/michael/ssd_cache/Lesion/train_logs', help='train logs location')
parser.add_argument('--GPU', type=str, default='0', help='specify which GPU to train on')
parser.add_argument('--debug', type=bool, default=False, help='set to debug mode')
options = parser.parse_args()



def train(rank, world_size, options): 
    
    torch.manual_seed(77777)
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    
    if options.DDP:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '77777'
        torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        is_first = rank==0
    else:
        is_first = True
    
    # initialize data
    train_labaled_dataset = data.DataSet(source_file_name='labeled_train.h5')
    train_labaled_sampler = torch.utils.data.RandomSampler(train_labaled_dataset, replacement=True, num_samples=options.epoch_samples)
    train_labaled_loader = data.FastDataLoader(dataset=train_labaled_dataset, batch_size=options.batch_size,
                                               sampler=train_labaled_sampler, num_workers=4, pin_memory=True, drop_last=True, 
                                               prefetch_factor=8, persistent_workers=False)
    
    train_dataset = train_labaled_dataset
    train_loader = train_labaled_loader
    test_dataset = data.DataSet(source_file_name='test_data.h5')
    
    # initialize model 
    if options.fine_tune_last: 
        #options.max_lr = 1e-4 if options.max_lr >= 1e-3 else options.max_lr
        datetime_regex = re.compile(r'[a-zA-Z_]_(\d{4}.*)$')
        logs = [(foldername, datetime_regex.search(foldername).groups()[0]) for foldername in os.listdir(options.log_path)]
        logs = [(foldername, datetime.strptime(timestamp, '%Y_%b_%d_%p_%I_%M_%S')) for foldername, timestamp in logs]
        logs.sort(key=lambda item: item[1], reverse=True)
        last_log = logs[0][0]
        # load the pickle object
        with open(f'{options.log_path}/{last_log}/model.pickle', 'rb') as filein:
            model = pickle.load(filein)
        print(f'loading check point {last_log}/check_point.pth')
        # load last saved state dicts and setup loss function 
        epoch_start = model.load_check_point(f'{options.log_path}/{last_log}/check_point.pth')
        # change log_path
        log_path = f'{options.log_path}/{last_log}'
    elif options.model_pickle_path: 
        with open(options.model_pickle_path, 'rb') as filein:
            model = pickle.load(filein)
        epoch_start = 0
    else:
        model = classification_cuda(options=options, input_shape=(patch_size, patch_size))
        epoch_start = 0 
        
    model.update_rules(options)
    model.update_scheduler(options, epoch_start)
    
    if is_first:
        # initialize new folder for logs, only do this when folder not exist 
        if not options.fine_tune_last: 
            time = datetime.now().strftime('%Y_%b_%d_%p_%I_%M_%S')
            log_path = f'{options.log_path}/{model.name}_{time}'
            os.mkdir(log_path)
            [os.mkdir(f'{log_path}/{foldername}') for foldername in ['models', 'test_images']]
            shutil.copytree(os.getcwd(), f'{log_path}/code_used')
            with open(f'{log_path}/train_epoch_log.csv', 'w') as fileout: 
                fileout.write(','.join(['epoch'] + 
                                    [f'train_{name}' for name in model.loss_names] + 
                                    [f'test_{name}' for name in model.loss_names]
                                    )
                              + '\n')
        # dump model and options 
        with open(f'{log_path}/model.pickle', 'wb') as fileout: 
            pickle.dump(model, fileout)
        with open(f'{log_path}/options.pickle', 'wb') as fileout:
            pickle.dump(options, fileout)
        
        print(model)
        print('-'*line_width)
        print(f'{len(train_dataset):_} images, {options.epoch_samples:_} samples per epoch, '
              f'{len(train_dataset) // options.epoch_samples} epochs for full coverage')
        print(f'{model.name} parameters: {model.compute_params()}')
        print(f'foldername: {log_path}')
    
    max_batch = len(train_loader)
    current_lr = options.max_lr
    epoch_start += 1 
    epoch_max = epoch_start + options.max_epoch
    # start training 
    for epoch_current in range(epoch_start, epoch_max): 
        
        start_time = datetime.now() 
        model.set_to_train()
        print('-'*line_width) if is_first else None 
        # do batches 
        for batch_index, input_data in enumerate(train_loader, 1):
            
            # this should not be changed, modification of loss and input should be done in model methods 
            model.set_input(input_data)
            current_lr = model.train(batch_index)
            
                    # dynamic alignment by length
            line = f'[{epoch_current:>{len(str(epoch_max))}}/{epoch_max-1}]' + \
                   f'[{batch_index:>{len(str(max_batch))}}/{max_batch}] ' + \
                   f'current learning rate: {current_lr:5e}'
            print(f'\r{options.schedule}{line}', end=' ', flush=True) if is_first else None 
        
        # exit training, do test 
        if is_first:
            os.mkdir(f'{log_path}/test_images/epoch_{epoch_current}')
            model.set_to_test()
            epoch_test = [] 
            with torch.no_grad(): 
                for image_index, input_data in enumerate(test_dataset): 
                    input_data = torch.unsqueeze(input_data, 0)
                    model.set_input(input_data)
                    
                    # prep the data
                    prediction, target, images = model.test()
                    
                    epoch_test.append([prediction, target])
                    # flush image to disk 
                    cv2.imwrite(f'{log_path}/test_images/epoch_{epoch_current}/T{target:.0f}_P{prediction:.2f}_{test_dataset.name}.jpg', images)
            
            # we're done with doing test, write to log file 
            train_loss = [f'{value:.5f}' for value in model.gather_loss('train')]
            test_loss =  [f'{value:.5f}' for value in model.gather_loss('test', epoch_test)]
        
            with open(f'{log_path}/train_epoch_log.csv', 'a') as fileout: 
                fileout.write(','.join([str(epoch_current)] + train_loss + test_loss) + '\n')
                
            # save model state_dict and check_point
            model.save_inference(f'{log_path}/models/epoch_{epoch_current}.pth')
            model.save_check_point(f'{log_path}/check_point.pth', epoch=epoch_current)
            
            # use df.to_string() for convenience 
            epoch_df = pd.DataFrame(columns=model.loss_names, data=[train_loss, test_loss])
            epoch_df.index = ['train', 'test']
            time_used = datetime.now() - start_time
            time_str = f'epoch time: {time_used.seconds // 60 :>02}:{time_used.seconds % 60 :>02}'
            print(time_str, epoch_df.to_string(), sep='\n')
        
        # exit validation phase  
    # exit training 
    
    if options.DDP:
        torch.distributed.destroy_process_group()
    
    return None 

    
if __name__ == '__main__':
    
    options.schedule = ''
    
    if options.preprocess:
        data.make_model_data() 

    if 'win' in sys.platform.lower():
        options.log_path = 'Z:/Data/Lesion/train_logs'
    
    if options.debug: 
        torch.autograd.set_detect_anomaly(True)
        options.max_epoch = 2
        options.epoch_samples = 40
        options.epoch_updates = 10
        options.batch_size = 4 
    else:
        warnings.filterwarnings("ignore")

    if len(options.GPU) > 1:
        world_size = len(options.GPU.split(','))
        options.DDP = True 
        torch.multiprocessing.spawn(train, nprocs=world_size, args=(world_size, options))
    else:
        options.DDP = False
        train(options.GPU, 0, options)

