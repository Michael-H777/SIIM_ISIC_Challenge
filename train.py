import data 
from packages import * 
from configs import * 

from loss import * 
from models.unet import * 
from models.pix2pix import  * 


parser = argparse.ArgumentParser(description='templet train file')
parser.add_argument('--preprocess', type=bool, default=False, help='process training data using specified directory')
parser.add_argument('--start_lr', type=float, default=1e-2, help='initial learning rate, decay to end_learning_rate with cosine annihilation')
parser.add_argument('--end_lr', type=float, default=1e-5, help='ending learning rate')
parser.add_argument('--max_epoch', type=int, default=50, help='maximum epoch for training')
parser.add_argument('--epoch_samples', type=int, default=5_000, help='samples for each epochs')
parser.add_argument('--epoch_updates', type=int, default=100, help='allowed updates for each epochs')
parser.add_argument('--batch_size', type=int, default=8, help='batch size when training')
parser.add_argument('--fine_tune_last', type=bool, default=False, help='fine tune last model?')
parser.add_argument('--model_pickle_path', type=str, default='', help='pickled model path, use with scheduler')
parser.add_argument('--log_path', type=str, default='/home/michael/ssd_cache/SMART/train_logs', help='train logs location')
parser.add_argument('--task_type', type=str, default='Denoise', help='Task type in train_log_path folder')
parser.add_argument('--GPU', type=str, default='0', help='specify which GPU to train on')
parser.add_argument('--debug', type=bool, default=False, help='set to debug mode')
options = parser.parse_args()


def train(rank, world_size, options): 
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = options.GPU
    options.log_path = f'{options.log_path}/{options.task_type}'
    #torch.backends.cudnn.benchmark = True
    
    # initialize data
    train_dataset = data.DataSet(source_file_name='labeled_train.h5')
    train_sampler = torch.utils.data.RandomSampler(train_dataset, replacement=True, num_samples=options.epoch_samples)
    train_loader = data.FastDataLoader(dataset=train_dataset, batch_size=options.batch_size,
                                       sampler=train_sampler, num_workers=4, pin_memory=True, drop_last=True, 
                                       prefetch_factor=6, persistent_workers=True)
    test_dataset = data.DataSet(source_file_name='test_data.h5')
    
    # initialize model 
    if options.fine_tune_last: 
        datetime_regex = re.compile(r'[a-zA-Z_]_(\d{4}.*)$')
        logs = [(foldername, datetime_regex.search(foldername).groups()[0]) for foldername in os.listdir(options.log_path)]
        logs = [(foldername, datetime.strptime(timestamp, '%Y_%b_%d_%p_%I_%M_%S')) for foldername, timestamp in logs]
        logs.sort(key=lambda item: item[1], reverse=True)
        last_log = logs[0][0]
        # load the pickle object
        with open(f'{options.log_path}/{last_log}/model.pickle', 'rb') as filein:
            model = pickle.load(filein)
        # load last saved state dicts and setup loss function 
        check_point = torch.load(f'{options.log_path}/{last_log}/check_point.pth')
        # idk if this is necessary, but it make sense 
        model.model = torch.nn.DataParallel(model.model).cuda()
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.model.eval()
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.make_loss_func(check_point['loss'])

        # change log_path
        log_path = f'{options.log_path}/{last_log}'
        epoch_start = checkpoint['epoch'] + 1
    elif options.model_pickle_path: 
        with open(f'{options.log_path}/{options.model_pickle_path}/model.pickle', 'rb') as filein:
            model = pickle.load(filein)
        epoch_start = 0
    else:
        model = Pix2Pix_cuda(name='NLD_Pix2Pix')
        epoch_start = 0 

    model.update_options(options)
    # distribute to GPU
    if options.DDP: 
        pass
    
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
                          +'\n')
    # dump model and options 
    with open(f'{log_path}/model.pickle', 'wb') as fileout: 
        pickle.dump(model, fileout)
    with open(f'{log_path}/options.pickle', 'wb') as fileout:
        pickle.dump(options, fileout)
        
    print(f'{len(train_dataset):_} images, {options.epoch_samples:_} samples per epoch, '
          f'{len(train_dataset) // options.epoch_samples} epochs for full coverage')
    print(f'{model.name} parameters: {model.compute_params()}')
    print(f'foldername: {log_path}')
    max_batch = len(train_loader)
    # start training 
    for epoch_current in range(epoch_start, epoch_start + options.max_epoch): 
        start_time = datetime.now() 
        current_lr = model.update_lr(epoch_current)
        print('-'*line_width)
        # do batches 
        for batch_index, input_data in enumerate(train_loader, 1):
            
            # this should not be changed, modification of loss and input should be done in model methods 
            model.set_input(input_data)
            model.take_step(batch_index)
            
                    # dynamic alignment by length
            line = f'[{epoch_current:>{len(str(options.max_epoch))}}/{options.max_epoch}]' + \
                   f'[{batch_index:>{len(str(max_batch))}}/{max_batch}] ' + \
                   f'lr: {current_lr:.5e}'
            print(f'\r{line}', end=' ', flush=True)
        
        os.mkdir(f'{log_path}/test_images/epoch_{epoch_current}')
        # exit training, do test 
        with torch.no_grad(): 
            for image_index, input_data in enumerate(test_dataset): 
                input_data = torch.unsqueeze(input_data, 0)
                model.set_input(input_data)
                model.validation()
                
                # prep the data
                input_data = model.input_data[0,0,:,:].cpu().numpy()
                prediction = model.prediction[0,0,:,:].cpu().numpy() 
                target = model.target[0,0,:,:].cpu().numpy()
                
                # flush image to disk 
                gap = np.zeros((input_data.shape[-1], 20))
                image = np.concatenate([input_data, gap, prediction, gap, target], axis=1)
                image = np.float32(image)
                tifffile.imwrite(f'{log_path}/test_images/epoch_{epoch_current}/{test_dataset.name}.tiff', image)
        
        # exit torch.no_grad() 

        # we're done with doing test, write to log file 
        train_loss = [f'{value:.5f}' for value in model.gather_train_loss()]
        test_loss =  [f'{value:.5f}' for value in model.gather_test_loss()]
        with open(f'{log_path}/train_epoch_log.csv', 'a') as fileout: 
            fileout.write(','.join([str(epoch_current)] + train_loss + test_loss) + '\n')
        
        # save model state_dict and check_point
        model.save_model(f'{log_path}/models')
        model.save_check_point(log_path)
        
        # use df.to_string() for convenience 
        epoch_df = pd.DataFrame(columns=model.loss_names, data=[train_loss, test_loss])
        epoch_df.index = ['train', 'test']
        time_used = datetime.now() - start_time
        time_str = f'epoch time: {time_used.seconds // 60 :>02}:{time_used.seconds % 60:>02}'
        print(time_str, epoch_df.to_string(), sep='\n')
    # exit training 
    
    if options.DDP:
        pass
    
    return None 


def main(): 

    if 'win' in sys.platform.lower():
        options.log_path = '//192.168.1.4/SSD_Cache/SMART/train_logs'
        
    if options.debug: 
        options.max_epoch = 5
        options.epoch_samples = 40
        options.epoch_updates = 10
        options.batch_size = 4 
    
    if len(options.GPU) > 1:
        options.DDP = True 
    else:
        options.DDP = False
        train(0, 0, options)

    

    
if __name__ == '__main__':
    
    if options.preprocess:
        data.make_model_data() 
    
    main() 
    