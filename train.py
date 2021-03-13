import data 
from packages import * 
from configs import * 


from loss import * 
from models import * 

parser = argparse.ArgumentParser(description='templet train file')
parser.add_argument('start_learning_rate', type=float, default=1e-3, help='initial learning rate, decay to end_learning_rate with cosine annihilation')
parser.add_argument('end_learning_rate', type=float, default=1e-5, help='ending learning rate')
parser.add_argument('max_epoch', type=int, default=50, help='maximum epoch for training')
parser.add_argument('epoch_samples', type=int, default=10_000, help='samples for each epochs')
parser.add_argument('epoch_updates', type=int, default=100, help='allowed updates for each epochs')
parser.add_argument('batch_size', type=int, default=128, help='batch size when training')
parser.add_argument('fine_tune_last', type=bool, default=False, help='fine tune last model?')
parser.add_argument('model_pickle_path', type=str, default='', help='pickled model path, use with scheduler')
parser.add_argument('log_path', type=str, default='/home/michael/ssd_storage/SMART/train_logs', help='train logs location')
parser.add_argument('task_type', type=str, default='Denoise', help='Task type in train_log_path folder')
parser.add_argument('GPUs', type=str, default='0,1,2,3', help='specify which GPU to train on')
options = parser.parse_args()

parser.log_path = f'{options.log_path}/{options.task_type}'


def train(options, rank): 
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    
    # initialize data
    train_dataset = data.dataset(source_file_name='labeled_train.h5')
    train_sampler = torch.utils.data.RandomSampler(train_dataset, replacement=True, num_samples=options.epoch_samples)
    train_loader = data.fast_data_loader(dataset=train_dataset, batch_size=options.batch_size, shuffle=True, 
                                         sampler=train_sampler, num_workers=6, pin_memory=True, drop_last=True, 
                                         prefetch_factor=4, persistent_workers=True)
    test_dataset = data.dataset(source_file_name='test_data.h5')
    
    # initialize model, also change train `for` loop configs
    if options.fine_tune_last: 
        datetime_regex = re.compile(r'[a-zA-Z_]_(\d{4}.*)$')
        logs = [(foldername, datetime_regex.search(foldername).groups[0]) for foldername in os.listdir(options.log_path)]
        logs = [(foldername, datetime.strptime(timestamp, '%Y_%b_%d_%H_%M_%S')) for foldername, timestamp in logs]
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
    elif options.model_pickle_path: 
        with open(f'{options.log_path}/{options.model_pickle_path}/model.pickle', 'rb') as filein:
            model = pickle.load(filein)
    else:
        model = Pix2Pix.Pix2Pix(in_channels=1, out_channels=1)

    model.update_options(options)
    # distribute to GPU
    if options.DDP: 
        pass
        
    # initialize new folder for logs, only do this when folder not exist 
    if not options.fine_tune_last: 
        time = datetime.now.strftime('%Y_%b_%d_%H_%M_%S')
        log_path = f'{options.log_path}/{model.name}_{time}'
        os.mkdir(log_path)
        [os.mkdir(f'{log_path}/{foldername}') for foldername in ['model', 'test_images']]
        shutil.copytree(os.getcwd(), f'{log_path}/{code_used}')
        with open(f'{log_path}/train_epoch_log.csv', 'w') as fileout: 
            fileout.write(','.join(['epoch'] + 
                                   [f'train_{name}' for name in model.loss_names] + 
                                   [f'test_{name}' for name in model.loss_names]
                                   )
                          +'\n')
        with open(f'{log_path}/model.pickle', 'wb') as fileout: 
            pickle.dump(model, fileout)
        
    print(f'{model.name} parameters: {model.compute_params()}')
    print(f'foldername: {log_path}')
    max_batch = len(train_loader)
    # start training 
    for epoch_current in range(options.max_epoch): 
        start_time = datetime.now() 
        current_lr = model.update_lr(epoch_current)
        print('-'*120)
        # do batches 
        for batch_index, data in enumerate(train_loader, 1):
            
            # this should not be changed, modification of loss and input should be done in model methods 
            model.set_input(data)
            model.forward() 
            model.take_step(batch_index)
            
                    # dynamic alignment by length
            line = f'[{epoch_current:>{len(str(options.max_epoch))}}/{options.max_epoch}]'
                   f'[{batch_index:>{len(str(max_batch))}}/{len(max_batch)}] '
                   f'lr: {current_lr}'
            print(f'\r{line:<120}', end='' if batch_index!=max_batch else '\n', flush=True)
        
        os.mkdir(f'{log_path}/{test_images}/{epoch_current}')
        # exit training, do test 
        with torch.no_grad(): 
            for image_index, data in enumerate(test_dataset): 
                model.set_input(data)
                model.forward()
                model.test()
                
                # prep the data
                input_data = model.input[0,0,:,:].cpu().numpy()
                prediction = model.prediction[0,0,:,:].cpu().numpy() 
                target = model.target[0,0,:,:].cpu().numpy()
                
                # flush image to disk 
                gap = np.zeros((20, data.shape[-1]))
                image = np.concatenate([input_data, gap, prediction, gap, target], axis=1)
                image = np.float32(image)
                tifffile.imwrite(f'{log_path}/test_images/{epoch_current}/{test_dataset.name}.tiff', image)
            
        # exit torch.no_grad() 

        # we're done with doing test, write to log file 
        train_loss = model.gather_train_loss() 
        test_loss = model.gather_test_loss() 
        with open(f'{log_path}/train_epoch_log.csv', 'w') as fileout: 
            fileout.write(','.join([str(epoch_current)] + train_loss + test_loss) + '\n')
        
        # save model state_dict and check_point
        model.save_model(f'{log_path}/model')
        model.save_check_point(log_path)
        
        # use df.to_string() for convenience 
        epoch_df = pd.DataFrame(columns=model.loss_names, data=[train_loss, test_loss])
        epoch_df.index = ['train', 'test']
        time_used = datetime.now() - start_time
        time_str = f'{time_used.seconds // 60 :>02}/{time_used.seconds:>02}'
        print(time_str, epoch_df.to_string(), sep='\n')
    # exit training 
    
    if options.DDP:
        pass
    
    return None 


def main(): 
    pass    
    
    
    
if __name__ == '__main__':
    main() 